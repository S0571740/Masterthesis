import os
import random
from SyncTokenRateLimiter import SyncTokenRateLimiter, get_llm_token_limiter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class RateLimitedChatOpenAI(ChatOpenAI):
    def __init__(self, token_limiter: SyncTokenRateLimiter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_limiter = token_limiter

    def _count_tokens(self, prompt: str) -> int:
        import tiktoken

        enc = tiktoken.encoding_for_model(self.model_name)
        return len(enc.encode(prompt))

    def _call(self, prompt: str, stop=None):
        tokens_needed = self._count_tokens(prompt) + (self.max_tokens or 0)
        if self._token_limiter is not None:
            self._token_limiter.wait_for_tokens(tokens_needed)
        return super()._call(prompt, stop=stop)


class Evaluator:
    CRITERIA = {
        "comprehensiveness": """How much detail does the answer provide to cover all the aspects and details of the
        question? A comprehensive answer should be thorough and complete, without being redundant or irrelevant.
        For example, if the question is ’What are the benefits and drawbacks of nuclear energy?’, a comprehensive
        answer would provide both the positive and negative aspects of nuclear energy, such as its efficiency,
        environmental impact, safety, cost, etc. A comprehensive answer should not leave out any important points
        or provide irrelevant information. For example, an incomplete answer would only provide the benefits of
        nuclear energy without describing the drawbacks, or a redundant answer would repeat the same information
        multiple times.""",
        "diversity": """How varied and rich is the answer in providing different perspectives and insights
        on the question? A diverse answer should be multi-faceted and multi-dimensional, offering different
        viewpoints and angles on the question. For example, if the question is ’What are the causes and effects
        of climate change?’, a diverse answer would provide different causes and effects of climate change, such
        as greenhouse gas emissions, deforestation, natural disasters, biodiversity loss, etc. A diverse answer
        should also provide different sources and evidence to support the answer. For example, a single-source
        answer would only cite one source or evidence, or a biased answer would only provide one perspective or
        opinion.""",
        "directness": """How specifically and clearly does the answer address the question? A direct answer should
        provide a clear and concise answer to the question. For example, if the question is ’What is the capital
        of France?’, a direct answer would be ’Paris’. A direct answer should not provide any irrelevant or
        unnecessary information that does not answer the question. For example, an indirect answer would be ’The
        capital of France is located on the river Seine’.""",
        "empowerment": """How well does the answer help the reader understand and make informed judgements about
        the topic without being misled or making fallacious assumptions. Evaluate each answer on the quality of
        answer as it relates to clearly explaining and providing reasoning and sources behind the claims in the
        answer.""",
    }

    def __init__(
        self, key, root_dir="", llm_model="gpt-4.1-nano", output_dir="evaluation_output", locality="local_2"
    ):
        self.root_dir = root_dir
        self.locality = locality
        self.questions_dir = os.path.join(root_dir, "Questions")
        self.ms_dir = os.path.join(root_dir, "Answers", locality, "ms")
        self.neo_dir = os.path.join(root_dir, "Answers", locality, "neo")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.llm = RateLimitedChatOpenAI(
            model=llm_model,
            api_key=key,
            token_limiter=get_llm_token_limiter(),
            max_completion_tokens=8000,
        )
        self.output_parser = StrOutputParser()

    def build_task_chain(self, criteria):
        """Build a reusable chain for evaluation."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert evaluator."),
                ("human", "{evaluation_task}"),
            ]
        )
        return prompt | self.llm | self.output_parser

    def _get_answer_file(self, system_dir, question_file, index, system_tag):
        question_name = os.path.splitext(question_file)[0]
        ans_file = f"response_{self.locality}_{system_tag}_{index+1}.txt"
        return os.path.join(system_dir, question_name, ans_file)

    def iter_data(self):
        """Same as your current version."""
        for q_file in sorted(os.listdir(self.questions_dir)):
            q_path = os.path.join(self.questions_dir, q_file)
            if not q_file.endswith(".txt"):
                continue

            with open(q_path, "r", encoding="utf-8") as f:
                questions = [line.strip() for line in f if line.strip()]

            for idx, question in enumerate(questions):
                ms_path = self._get_answer_file(self.ms_dir, q_file, idx, "ms")
                neo_path = self._get_answer_file(self.neo_dir, q_file, idx, "neo")

                with open(ms_path, "r", encoding="utf-8") as f_ms:
                    ms_answer = f_ms.read().strip()
                with open(neo_path, "r", encoding="utf-8") as f_neo:
                    neo_answer = f_neo.read().strip()

                yield question, ms_answer, neo_answer

    def map_back(self, llm_eval, shuffle_map):
        mapped_eval = llm_eval.lower()
        for resp, system in shuffle_map.items():
            mapped_eval = mapped_eval.replace(resp, system.upper())
        return mapped_eval

    def run_llm(self, prompt):
        response = self.client.chat.completions.create(
            model=self.llm_model,  # model_name is already used internally by ChatOpenAI
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def evaluate_all(self):
        question_counter = 1

        for question, ms_ans, neo_ans in self.iter_data():
            responses = [("ms", ms_ans), ("neo", neo_ans)]

            for criteria_key, criteria_text in self.CRITERIA.items():
                shuffled = random.sample(responses, len(responses))
                shuffle_map = {
                    f"answer {i+1}": system.lower()
                    for i, (system, _) in enumerate(shuffled)
                }

                # Build task string (human message content)
                eval_task, _ = self.build_task_prompt(question, shuffled, criteria_text)

                # Get chain
                chain = self.build_task_chain(criteria_text)

                # Run evaluation
                llm_eval = chain.invoke({"evaluation_task": eval_task})

                mapped_eval = self.map_back(llm_eval, shuffle_map)

                filename = os.path.join(
                    self.output_dir, f"eval_{question_counter}_{criteria_key}.txt"
                )
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"{criteria_key.upper()}:{question}\n\n")
                    f.write(mapped_eval + "\n")

                yield question, ms_ans, neo_ans, criteria_key, mapped_eval
            question_counter += 1

    def build_task_prompt(self, question, answers, criteria):
        shuffled = random.sample(answers, len(answers))
        shuffle_map = {
            f"answer {i+1}": system for i, (system, _) in enumerate(shuffled)
        }
        answer_texts = [ans for _, ans in shuffled]

        prompt = f"""
---Role---
You are a helpful assistant responsible for grading two answers to a question that are provided by two
different people.
---Goal---
Given a question and two answers (Answer 1 and Answer 2), assess which answer is better according to
the following measure:
{criteria}
Your assessment should include two parts:
- Winner: either 1 (if Answer 1 is better) and 2 (if Answer 2 is better) or 0 if they are fundamentally
similar and the differences are immaterial.
- Reasoning: a short explanation of why you chose the winner with respect to the measure described above.
Format your response as a JSON object with the following structure:
{{
"winner": <1, 2, or 0>,
"reasoning": "Answer 1 is better because <your reasoning>."
}}
---Question---
{question}
---Answer 1---
{answer_texts[0]}
---Answer 2---
{answer_texts[1]}
Assess which answer is better according to the following measure:
{criteria}
Output:
""".strip()

        return prompt, shuffle_map
