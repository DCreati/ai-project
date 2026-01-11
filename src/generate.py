from typing import List, Tuple
from dataclasses import dataclass
import logging
import string

import spacy
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput # .generate(...) output. Used for code hints
from transformers.generation.utils import GenerationMixin # You can check the code with GenerationMixin.generate
from transformers import PreTrainedModel, LlamaTokenizer # Used for code hints

from retriever import BM25

DEBUG = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

@dataclass
class Block:
    text: str = None
    tokens: List[str] = None # The choice of saving tokens instead of ids is due to the ease of handling “▁” when merging words.
    range_: List[Tuple[int, int]] = None # A combined unit is noted as a word. here the intervals of each word are noted (left closed, right open)
    @property
    def len_tokens(self):
        return len(self.tokens)
    @property
    def len_words(self):
        return len(self.range_)

def merge_blocks(blocks: List[Block]) -> Block:
    text = "".join([block.text for block in blocks])
    tokens = sum([block.tokens for block in blocks], [])
    range_ = []
    st = 0
    for block in blocks:
        if block.range_:
            for l, r in block.range_:
                range_.append((st+l, st+r))
            st = range_[-1][1]
    return Block(text=text, tokens=tokens, range_=range_)

class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }

@dataclass
class GeneratorOutput:
    '''
Return Value:
- ended: Whether the end is detected. Determined by eos_token
- blocks: Chunks of text segmentation storage. In later use, this will be based on [demo, “Question:”, question, “\nAnswer:”, text],
    and the newly generated new_text division. 
    The merge word operation will be done within each block. 
    The convention is that the merge word will not cross the junction of two segments.
- atten: (len_words, len_new_words)。\
    Already averaged over multiple heads. Note that it is under the sense after the word merge.
- max_atten: (len_new_words,)
- entropies: (len_new_words,)

Note: The source code stores the results of merging words as strings and separates the word columns with spaces when converting them to strings。\
    I don't think this is good because it puts a stronger presupposition on the behavior of the splitter.
- On the one hand there are a lot of things that are stored as escapes when encoded by the participle encoder, such as “ ‘, ’\\\n”, “\\\t” becoming “▁”, “<0x0A>”, “<0x09>” respectively (and there are many more I'm not sure of).\
    Merging them into a new string tends to confuse the splitter about what it originally meant, and it won't turn back.
- On the other hand, there is no guarantee that the merged words are separated by spaces. For example, there is no space between “\\n” and the next word.
    '''
    ended: bool
    blocks: List[Block] = None
    merged_blocks: Block = None
    atten: Tensor = None
    max_atten: Tensor = None
    entropies: Tensor = None
    @property
    def new_text(self):
        return self.blocks[-1].text
    @property
    def len_new_words(self):
        return self.blocks[-1].len_words
    
class Generator:
    def __init__(
        self,
        model_name_or_path: str
    ):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # self.model_config = AutoConfig.from_pretrained(model_name_or_path)
        # assert self.model_config.model_type == "llama"
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto") # device_map means automatically split to the graphics card
        self.model: PreTrainedModel
        logger.info(f"device = {self.model.device}")
        # Anyway only referenced llama
        self.space_token = "▁"
        self.tokenizer.pad_token = self.tokenizer.eos_token # Set the fill marker to the end marker
        # Looking at llama's tokenizer_config shows that its bos_token is “<s>”, eos_token is “</s>”, and pad_token is null.
        
        self.tokens_cannot_merged = {
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode("0" + ch)[-1:])[0]
            # The “0” is to prevent the creation of a combination with “▁”.
            for ch in string.whitespace + string.punctuation
        } | {self.space_token, self.tokenizer.bos_token, self.tokenizer.eos_token}

    def simply_generate(
        self,
        input_text: str,
        max_length: int
    ) -> Tuple[bool, str]:
        '''
        return ended, new_text
        '''
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device) # (batch_size=1, input_length)
        input_length = input_ids.shape[1]
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            stop_strings = "\n",
            tokenizer=self.tokenizer
        )[0, input_length:]
        if output_ids.shape[0] == 0:
            # I don't think that's the case. output_ids should at least output eos_token.
            logger.info("generate '' in simply_generate()!")
            return True, ""
        if output_ids[0] == self.tokenizer.bos_token_id:
            output_ids = output_ids[1:]
        if output_ids[-1] == self.tokenizer.eos_token_id:
            return True, self.tokenizer.decode(output_ids[:-1])
        return False, self.tokenizer.decode(output_ids)

    def tokenize(
        self,        
        text: str,
        is_start: bool = False # If not, delete bos_token
    ):
        ids = self.tokenizer.encode(text) # List[int]
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if len(tokens) > 0:
            if not is_start and tokens[0] == self.tokenizer.bos_token:
                tokens = tokens[1:]
        
        return tokens
        
    def merge_tokens(
        self,
        tokens
    ) -> List[Tuple[int, int]]:
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) \
                or tokens[i] in self.tokens_cannot_merged \
                or tokens[i-1] in self.tokens_cannot_merged:
                range_.append([i, i+1]) # as a neologism
            else:
                range_[-1][1] += 1
        return range_

    def build_block(
        self,        
        text: str,
        is_start: bool = False # If not, delete bos_token
    ) -> Block:
        tokens = self.tokenize(text, is_start=is_start)
        range_ = self.merge_tokens(tokens)
        return Block(text=text, tokens=tokens, range_=range_)

    def generate(
        self,
        input_texts: List[str], # The inputs are already segmented, and in the later implementation are：[demo, "\nQuestion:", question, "\nAnswer:", text]
        max_length: int,
    ) -> GeneratorOutput:
        blocks = []
        for text in input_texts:
            blocks.append(self.build_block(text, is_start=not blocks))

        input_tokens = sum([block.tokens for block in blocks], [])
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_tokens)], device=self.model.device)
        input_len_tokens = len(input_tokens)

        # if DEBUG:
        #     print("The input_text to be used for initial generation:")
        #     print(self.tokenizer.convert_tokens_to_string(input_tokens))
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            return_dict_in_generate=True,
            output_scores=True, # scores: Tuple[Tensor(batch_size, vocab_size)] len(scores) == generated_length
            # output_attentions=True
            stop_strings="\n",
            tokenizer=self.tokenizer
        )
        outputs: GenerateDecoderOnlyOutput

        tokens = self.tokenizer.convert_ids_to_tokens(outputs.sequences[0, input_len_tokens:]) # List[str]
        ended = (tokens[-1] == self.tokenizer.eos_token)
        if ended:
            tokens = tokens[:-1]
            # Note: If tokens have “</s>” in them (“<s>” is the same), it will still be retained when converted to a string. So delete it here.
        text = self.tokenizer.convert_tokens_to_string(tokens)
        range_ = self.merge_tokens(tokens)
        new_block = Block(text=text, tokens=tokens, range_=range_)

        blocks.append(new_block)
        merged_blocks = merge_blocks(blocks)

        # These below are quite different from the source code.
        # - The source code is a separate set of attentions for new_tokens, here we try to intercept and normalize the new_tokens part based on the original attentions.
        # - In the source code, the operation of maximizing the attention comes before the attention in the sense of merging, and vice versa here. Here, the attentions in the merge sense are sought at the beginning.
        # - The source code is to maximize ATTENTION before averaging over multiple heads, here it is the other way around.

        atten = self.model(outputs.sequences, output_attentions=True).attentions[-1][0][:, -new_block.len_tokens:, :] # (num_heads, new_len_tokens, len_tokens)
        # Because the output format of generate's attentions is weird, lazy recalculation of attentions.
        # Theoretically organizing the formatting and making up the last token's pair of previous attentions would work too.
        # outputs.attentions: Tuple[Tuple[Tensor(batch_size, num_heads, generated_length, sequence_length)]]
        # Outermost generated_length, second outermost num_layers
        # For the first token generated, generated_length==input_len_tokens; other generated_length==1
        # For each token generated, sequence_length==the length of the input sequence before generating it

        # Attention in the combined sense:
        atten = atten.mean(dim=0)
        atten = torch.stack([atten[:, l:r].sum(dim=-1) for l, r in merged_blocks.range_], dim=-1) # For the same attentee (token), sum the attentions of the same group of attentees (token)
        atten = torch.stack([atten[l:r, :].mean(dim=-2) for l, r in range_], dim=-2)  # Attention of the same group of noticers (token) is averaged over the ATTENTION of the same notee (word).
        # assert atten.shape == (new_block.len_words, merged_blocks.len_words)

        # Maximum attention to be noticed
        atten_to_new = atten[:, -new_block.len_words:]
        atten_to_new /= atten.sum(dim=-1,keepdim=True) + 1e-10 # 归一化
        max_atten, _ = atten_to_new.max(dim=1)
        # assert max_atten.shape == (new_block.len_words,)

        # entropy (physics)
        probs = torch.stack(outputs.scores).softmax(dim=-1) # (new_len_tokens, batch_size, vocab_size)
        entropies = (-probs * torch.log(probs + 1e-10)).sum(dim=-1) # (new_len_tokens, batch_size)
        # if DEBUG:
        #     print("Printing of candidate words and entropies before merging：")
        #     for i, token in enumerate(tokens):
        #         print(f"{i}th {token}：entropy={entropies[i][0]} Candidates:", end="")
        #         li = [(probs[i][0][id].item(), self.tokenizer._convert_id_to_token(id)) for id in range(probs.shape[-1]) if probs[i][0][id] > 1e-6]
        #         li.sort(reverse=True)
        #         for j, (p, t) in enumerate(li):
        #             if p < 1e-6 or j >= 10:
        #                 break
        #             print(f"({t},{p})", end=" ")
        #         print()
        entropies = torch.stack([entropies[l:r, 0].sum() for l, r in range_])
        # assert entropies.shape == (new_block.len_words,)
        # The source code is averaging, the summing is used here.
        
        return GeneratorOutput(
            ended=ended,
            blocks=blocks,
            merged_blocks=merged_blocks,
            atten=atten,
            max_atten=max_atten,
            entropies=entropies
        )

def join_if_nonempty(*li, sep=" "):
    return sep.join([s for s in li if len(s) > 0])

def match(word: str, real_words): # Existence of real words as substrings of word
    for real_word in real_words:
        if real_word in word: # Determine if it is a substring
            return True
    return False

def get_top_sentence(text):
    prev = ""
    for sent in nlp(text).sents:
        prev += sent.text
        sent = sent.text.strip()
        if len(sent) > 0:
            return prev
    return ""

@dataclass
class CheckerOutput:
    '''
- hallucination: Is there a hallucination
- curr_st: Starting position of the hallucination sentence
- curr_en: Termination position of hallucinatory sentences
- curr_thres: Whether or not the words of the hallucinated sentence reach thresholds
    '''
    hallucination: bool 
    curr_st: int = None
    curr_en: int = None
    curr_thres: List[bool] = None

class DRAGIN:
    def __init__(self, args):
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        self.generator = Generator(self.model_name_or_path)
        self.tokenizer = self.generator.tokenizer
        self.retriever = BM25("wiki" if "es_index_name" not in args else self.es_index_name)
        self.counter = Counter()

    def hallucination_check(
        self,
        outputs: GeneratorOutput
    ) -> CheckerOutput: # Hallucination Detection
        # One problem with the implementation here is that scapy is called for clause splitting, but I don't know how much scapy's splitting is worse than LlamaTokenizer.
        # It would be nice to build on the clauses from the previous subjunctive.
        if DEBUG:
            print("Start detecting hallucinations.")
        new_block = outputs.blocks[-1]
        sentences = [sent.text.strip() for sent in nlp(new_block.text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        if DEBUG:
            print("The result of the clause obtained by calling spacy: ")
            for i, sent in enumerate(sentences):
                print(f"sentences{i}：{sent}")
        wid = 0
        for sid, sent in enumerate(sentences):
            wl, wr = wid, wid
            if wid == new_block.len_words:
                break
            while wr < new_block.len_words and sent not in self.tokenizer.convert_tokens_to_string(
                new_block.tokens[new_block.range_[wl][0]:new_block.range_[wr][1]]
            ):
                wr += 1
                # assert wr < new_block.len_words, "sent not in the remainder of new_text!"
                # This situation exists because of LlamaTokenizer's strange participles and spacy's strange breaks.
            if wr < new_block.len_words:
                wr += 1 # sent in words[wl, wr)
            wid = wr
            if wl == wr:
                continue
            if DEBUG:
                print("Current Sentence: ", self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[wl][0]:new_block.range_[wr-1][1]]), sep="\n")
            # Normalize by sentence and multiply by sentence length, consistent with the source code. Suitability to be explored.
            max_atten_sent = outputs.max_atten[wl: wr]
            max_atten_sent = max_atten_sent * (wr - wl) / (max_atten_sent.sum() + 1e-10)
            value = max_atten_sent * outputs.entropies[wl: wr]
            thres = (value > self.hallucination_threshold)
            if DEBUG:
                print("word|max_atten_sent|entropy|value|thres：")
                for i in range(wl, wr):
                    print(self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[i][0]:new_block.range_[i][1]]), 
                          max_atten_sent[i-wl].item(),
                          outputs.entropies[i-wl].item(),
                          value[i-wl].item(),
                          thres[i-wl].item(), sep="|")
            if True in thres:
                doc = nlp(sent)
                real_words = set(token.text for token in doc if token.pos_ in 
                    ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                for i in range(wl, wr):
                    tl, tr = new_block.range_[i]
                    word = self.tokenizer.convert_tokens_to_string(new_block.tokens[tl:tr])
                    if not match(word, real_words):
                        if DEBUG and thres[i-wl]:
                            print(f"Prefisso{i-wl} Numero：{self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[i][0]:new_block.range_[i][1]])} had reached the threshold, but the dummy word")
                        thres[i-wl] = False
                if True in thres:
                    return CheckerOutput(hallucination=True, curr_st=wl, curr_en=wr, curr_thres=thres)
            if DEBUG:
                print("No hallucinations were detected in the current sentence. Prepare the next sentence.")
        return CheckerOutput(hallucination=False)

    def generate_retrieve_qry(self, outputs: GeneratorOutput, check_info: CheckerOutput):
        # Memories: input_texts=[demo, "\nQuestion:", question, "\nAnswer:", text]
        # The prefixes in mind are in question and text+ptext, the keywords will only be selected here.

        ques_st = outputs.blocks[0].len_words + outputs.blocks[1].len_words
        ques_en = ques_st + outputs.blocks[2].len_words
        text_st = ques_en + outputs.blocks[3].len_words
        text_en = text_st + outputs.blocks[4].len_words + check_info.curr_st

        ques_atten = outputs.atten[check_info.curr_st:check_info.curr_en, ques_st:ques_en]
        text_atten = outputs.atten[check_info.curr_st:check_info.curr_en, text_st:text_en]
        
        ques_atten = ques_atten[check_info.curr_thres, :].sum(dim=0)
        text_atten = text_atten[check_info.curr_thres, :].sum(dim=0)

        doc = nlp(outputs.merged_blocks.text)
        real_words = set(token.text for token in doc if token.pos_ in 
            ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        real_pairs = []
        for i in range(ques_st, ques_en):
            a = ques_atten[i - ques_st]
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
            if match(word, real_words):
                real_pairs.append((a, word, i))
        for i in range(text_st, text_en):
            a = text_atten[i - text_st]
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
            if match(word, real_words):
                real_pairs.append((a, word, i))
        
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs.sort(key=lambda x: -x[0])
        real_pairs = real_pairs[:top_k]
        real_pairs.sort(key=lambda x: x[2])

        return " ".join([x[1] for x in real_pairs])

    def inference(self, question, demo, case):
        text = ""
        demo = "\n".join([d["case"] for d in demo]) # To avoid the model from generating a new Q&A, termination is attempted here.
        # exist WikiMultiHopQA case = f'Question: {question}\nAnswer:'. Therefore only QUESTION and not CASE is used later.
        if DEBUG:
            print("Ready to start reasoning")
        while True:
            old_len = len(text)
            # demo: description (including the FEWSHOT); text: the part that has been answered
            outputs = self.generator.generate(
                input_texts=[demo, "\nQuestion:", question, "\nAnswer:", text], 
                max_length=self.generate_max_length,
            )
            if DEBUG:
                print("Initially generate new text as: ", outputs.new_text, sep="\n")
            if self.use_counter == True:
                self.counter.add_generate(outputs.new_text, self.generator.tokenizer)
            if outputs.new_text.strip() == "":
                if DEBUG:
                    print("Detects that only blank characters have been generated and will interrupt generation.")
                break
            check_info = self.hallucination_check(outputs)
            if not check_info.hallucination:
                if DEBUG:
                    print("No hallucinations were detected.")
                text = join_if_nonempty(text, outputs.new_text.strip())
                if DEBUG:
                    print("Currently generated text: ", text, sep="\n")
                if outputs.ended or outputs.merged_blocks.len_tokens > self.generate_max_length:
                    if DEBUG:
                        if outputs.ended:
                            print("Terminator detected " if outputs.ended else " Detects that the text has reached its maximum length.")
                    break
            else:
                if DEBUG:
                    print("Hallucinations detected. Preparing for retrieval.")
                retrieve_qry = self.generate_retrieve_qry(outputs, check_info)
                if DEBUG:
                    print(f"retrieve_qry: {retrieve_qry}")
                # Unlike source code.
                # The source code does not directly take the data in the previous generate, but in addition to get a prefix string (question + text + ptext), recalculating the attention。
                # It's possible that it's to eliminate the effect of the demo (and the formatting part of the case outside of the QUESTION).
                # The ATTENTION, derived from the previous GENERATE section, is used here to extract the needed parts.
                # There is an option to align the normalization with the previous method, which is not used here for now.
                # Whether normalization is better is debatable; the question is to what extent the hallucination is affected by the FEW-shot.
                # If the attention of a word is focused on the FEW-shot part, then forcing the normalized post-attention value to rest in the restricted area might cause bias.

                docs = self.retriever(retrieve_qry, topk=self.retrieve_topk)
                self.counter.retrieve += 1
                if DEBUG:
                    print("Additional information was retrieved: ", docs, sep="\n")
                # Re-generate the new text:
                # Since ATTENTION is no longer involved here, it's lazy to ensure subjunctive consistency, but in theory it should work.
                # It may be more standardized to store the token as the base unit entirely, except for preprocessing, output results, and interactions with spacy and retrievers
                prompt = demo
                prompt += "\nContext:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                prompt += "Answer in the same format as before.\n"
                for i in [1, 2, 3]: # "Question:", question, "\nAnswer:"
                    prompt += outputs.blocks[i].text
                text = self.tokenizer.convert_tokens_to_string(
                    outputs.blocks[-2].tokens # text
                    + outputs.blocks[-1].tokens[:outputs.blocks[-1].range_[check_info.curr_st][0]] # ptext
                )
                prompt += text
                ended, new_texts = self.generator.simply_generate(
                    prompt, 
                    max_length=self.generate_max_length,
                )
                if self.use_counter == True:
                    self.counter.add_generate(new_texts, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = get_top_sentence(new_texts)
                # text += new_text
                text = join_if_nonempty(text, new_text.strip())
                if DEBUG:
                    print("Re-generate the new text: ", new_text, sep="\n")
                if DEBUG:
                    print("Currently generated text:", text, sep="\n")
                if ended and len(new_text) >= len(new_texts.strip()):
                    if DEBUG:
                        print("Terminator detected.")
                    break
                if len(self.tokenizer.encode(text)) > self.generate_max_length:
                    if DEBUG:
                        print("Detects that the text has reached its maximum length.")
                    break
            if old_len >= len(text): # That shouldn't be happening, should it?
                logger.info("old_len >= len(text) !")
                break
        if DEBUG:
            print("End of reasoning. Final text generated: ", text, sep="\n")
        return text
            