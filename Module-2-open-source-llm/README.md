# Week 2 notes

### Table of contents

- [2.1 Open-Source LLMs - Introduction](#21-open-source-llms---introduction)
    - [Open-Source LLMs](#open-source-llms)
    - [Replacing the LLM box in the RAG flow](#replacing-the-llm-box-in-the-rag-flow)
- [2.2 Using a GPU in Saturn Cloud](#22-using-a-gpu-in-saturn-cloud)
    - [Registering in Saturn Cloud](#registering-in-saturn-cloud)
    - [Configuring Secrets and Git](#configuring-secrets-and-git)
    - [Creating an instance with GPU](#creating-an-instance-with-gpu)
- [2.3 FLAN-T5-XL](#23-flan-t5)
    - [Hugging Face](#hugging-face)
    - [Google FlAN T5 XL](#google-flan-t5-xl)
    - [Rewriting the LLM Function](#rewriting-the-llm-function)
- [2.4 Phi 3 Mini](#24-phi-3-mini)
    - [Introduction to Microsoft's Phi3](#introduction-to-microsofts-phi3)
    - [Nvidia-SIM, Model Size and using Phi3](#nvidia-ami-model-size-and-using-phi3)
    - [Using Phi3 for RAG](#using-phi3-for-rag)
 - [2.5 Mistral-7B](#25-mistral-7b)
    - [Introduction to Mistral-7B](#introduction-to-mistral-7b)
    - [Using Mistral-7B](#using-mistral-7b)
    - [Saving, Loading the model and the LLM function](#saving-loading-the-model-and-the-llm-function)
 - [2.6 Exploring Open Source LLMs](#26-exploring-open-source-llms)
 - [2.7 Running LLMa locally without a GPU using Ollama](#27-running-llma-locally-without-a-gpu-using-ollama)
    - [Setting up and running Ollama](#setting-up-and-running-ollama)
    - [Replacement for OpenAI API](#replacement-for-openai-api)
    - [Running Ollama in Docker](#running-ollama-in-docker)
    - [Customise a model](#customize-a-model)

## 2.1 Open-Source LLMs - Introduction

In this week's lecture, we will be exploring alternatives to `OPENAI`, and in particular, we will be discussing more on **Open-Sourced LLMs** and how to run them. There are several ways to run open-source LLM models, depending on your technical expertise and available resources. Here are some options:

1. Local Machine:
- If you have a powerful computer with a good GPU, you can run models using libraries like Hugging Face's `Transformers` or `LlamaIndex`.
- This requires some technical knowledge and setup.

2. Cloud Services:
- Google Colab: Free option with GPU access, good for experimenting.
- AWS SageMaker, Google Cloud AI Platform, or Azure Machine Learning: For more robust, scalable solutions.

3. Specialized Platforms:
- `Hugging Face`: Offers a platform to run many open-source models.
- `Replicate`: Allows running various AI models in the cloud.

4. Self-hosted Solutions:
- Set up your own server or use a service like `Paperspace` or `Lambda Labs`.

5. Containerized Deployments:
- Use `Docker` to package and deploy models on various platforms.

6. Specialized Hardware:
- `Raspberry Pi` or other single-board computers for smaller models.

7. Open-source Frameworks:
- `Ollama`: Allows running LLMs locally with a simple interface.
- `LocalAI`: For running AI models on consumer-grade hardware.

8. Community-driven Platforms:
- `EleutherAI`: Offers access to some of their models.

Remember, running large models requires significant computational resources. Smaller models or quantized versions might be more suitable for personal use.

### Open-Source LLMs

There are many open-source LLM models available. Here's a list of some popular ones:

1. BERT (Bidirectional Encoder Representations from Transformers)
- Developed by Google
- Good for understanding context in language

2. GPT-2 (Generative Pre-trained Transformer 2)
- Created by OpenAI
- Smaller version of GPT-3, good for text generation

3. T5 (Text-to-Text Transfer Transformer)
- Developed by Google
- Versatile model for various NLP tasks

4. BLOOM (BigScience Large Open-science Open-access Multilingual Language Model)
- Created by BigScience
- Large multilingual model

5. LLaMA (Large Language Model Meta AI)
- Developed by Meta (Facebook)
- Range of sizes, from 7B to 65B parameters

6. Falcon
- Created by Technology Innovation Institute (TII)
- Known for efficiency and performance

7. RWKV
- An alternative architecture to Transformer models
- Good for both small and large-scale deployments

8. GPT-J and GPT-NeoX
- Developed by EleutherAI
- Open-source alternatives to GPT-3

9. OPT (Open Pre-trained Transformer)
- Released by Meta
- Designed to be more accessible for research

10. FLAN-T5
- Google's instruction-tuned version of T5

11. Pythia
- A suite of models from EleutherAI for studying AI behavior

13. Dolly
- Databricks' instruction-following model

These are to name a few and we should be having a feel of few of them in subsequent lectures. And what we are going to cover is how to run these models, which normally requires a lot of GPU. And we are going to need a proper environment for doing so. This too will be discussed as we progress into the course.

### Replacing the LLM box in the RAG flow

So what we are going to do is to replace the LLM "BOX" as seen below with some other open-sourced LLM. This would be the focus of this module.

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/abe89b5e-d322-44a5-a9ed-b22650d76cb2)

In particular, we will see how to set up an environment and use different LLMs, as well as accessing these models using `HuggingFace`.

## 2.2 Using a GPU in Saturn Cloud

In this section we are going to learn more about `SaturnCloud` and how to set up a GPU-enabled notebook as they are required for most open-sourced LLMs, and `SaturnCloud` provides the enviroment for doing so. There are of course alternatives like `GoogleColab` and `AWS SageMaker`, but feel free to use what works best for you.
 
### Registering in Saturn Cloud

Go to [SaturnCloud's](https://saturncloud.io/) website and click on `Get a Techinical Demo`

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/48979241-0e86-442b-b9fe-fb7efb4f94d2)

Next, in the landing page, input the details as follows - use your own personal email of course.

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/22a47b6e-e318-49a8-a52a-4af4f1086e1f)

### Configuring Secrets and Git

Navigate to `secrets` page through the naviagation panel on the left as follows:

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/9909010f-927f-48b9-85e2-c0c9fa509843)

The `secrets` page is for you to add your tokens. For example, your `OPENAI_API_KEY` or your `HuggingFace` token (we will look into `HuggingFace` in the next section).

To add a token is rather straightfoward, you just need to click on `New` on the top right-hand corner of the page, and then add the `Name` and `Value` of the token. Once done, click `Add`. It is very similar to the terminal command for adding an `environment variable` - `export OPENAI_API_KEY='xxx'`.

For setting up `git` access, you need to configure `USER SETTINGS` in Saturn Cloud as well as creating a secure connection by adding your `SSH public key` into your `Github` account.

1. Go to `USER SETTINGS`

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/a74d1278-83a1-4b34-88de-1981f21fe402)

2. Under the section `Git SSH Key`, click on `Create an SSH Key`
3. In the landing page, it should look like the following. You want to generate a public/private key pair within `SaturnCloud`, so click `Add`.

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/65408683-85ce-43ed-ac50-6f88a5df1ae2)

4. Now that you have generated your `SSH - public key`, you can add it to your git Host to create the secure connection. To do so you have to navigate to the `settings` page on your `Github` account, and then on the left panel select `SSH Key and GPG keys` as follows:

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/f991c984-4a7b-4876-9d14-7029d62304e3)

5. Next you have to click on `New SSH Key` - this should take you to `Add new SSH Key`, add the title as well as the `SSH - public key` you created in Saturn Cloud. You should have now created a secure connection

6. Once you’ve set up your SSH keys, you can add git repositories to your resources. From the left-hand menu, select the Git Repositories tab.

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/8c6f1212-0f41-478b-bf36-19625a5aec81)

7. From the git repositories page, select the `New` button at the top right corner. Here, you can add a repository via remote URL (this is the link you’d use when running `git clone` in the terminal).

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/735704c6-9377-46fe-8356-b90aeea31d67)

8. If you still feel a little lost, please read this [documentation](https://saturncloud.io/docs/using-saturn-cloud/gitrepo/#:~:text=Click%20on%20Create%20an%20SSH,the%20SSH%20key%20creation%20form.&text=After%20you%20have%20a%20key,to%20create%20the%20secure%20connection.)

### Creating an instance with GPU

Now that the `git` access is set up, we can now `Create a Jupyter Server`. To do so, follow these steps:

1. First navigate to the page for creating a `jupyter server` - click on `Resources`, then on `New Python Server`.
![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/716a286a-e637-4324-b9c5-65dad61b6a4a)
2. Under the `Overview` section, add any name you prefer but I went with `llm-zoomcamp-gpu`. For `Hardware`, make sure you select `GPU`, and the deafault size is as shown below (Note: its a free resource, so this is the only `GPU` resource that is available to us)
![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/8d4735aa-fec0-4a01-9f73-a6765c000ddb)
3. Next we need to choose our `image` as well as `pip install` extra packages as follows in the `Environment` section.
![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/29c43ae3-b377-47b0-a054-6189ca4cd8e9)
4. Since we added our `git repo` in the git repositories page earlier, we can now select it from the dropdown.
![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/ab64b443-7a0b-469b-b75a-6cee134cecd7)
5. That is pretty much it - we can now just click `create` at the bottom of the page.
6. This should lead you to the following page - if you need to add your `enironement variables` you can do so in the `Secrets and Roles` page as highlighted.
![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/7b53a264-38fa-42fc-9d92-3e6acf412686)

## 2.3 FLAN-T5

In this section we will be running our first open-source LLM `FLAN-T5` which is an advanced language model developed by Google AI. The model is available in `HuggingFace` library, and hence will be using `HuggingFace` platform extensively throughout this course.

### Hugging Face

At this point it would be good to describe a little bit more about Hugging Face and its importance in the AI community. Hugging Face is a game-changer in machine learning and natural language processing and a key agent in the democratization of AI. Thanks to transfer learning, it is playing a crucial role in making AI more accessible.

Simply put its the `Github` of the ML world - a collaborative platform brimming with tools that empower anyone to create, train, and deploy NLP and ML models using open-source code. And its main component is the `transforemers` library, which is why we had `pip install transformers` when setting up our Jupyter Server in Saturn Cloud. In a nutshell, the `transformers` library in Hugging Face allows easy access to all the open-source LLM models that we would want to use for this course and more. Not only that, it provides tools such as `pipeline` API as well as `tokenizers` to simplfy the process of fine-tuning pretrained models as well as making them more efficient for things like text processing.

For a more detailed introduction to `HuggingFace` and the `transformers` library, I would recommend to read this [article](https://www.datacamp.com/tutorial/what-is-hugging-face).

### Google FlAN T5 XL

The Google `FLAN-T5-XL` is a larger variant of the `FLAN-T5` model family. The model as well as their description and usage, amongst others, are available [here](https://huggingface.co/google/flan-t5-xl). And as mentioned eariler, its available on the `Hugging Face Model Hub`. 

The `FLAN-T5-XL` is a versatile model for a range of natural language processing tasks which includes Text Summarisation, Language Translation and Text Generation to name a few. Let's take the example usage from the model page shared earlier or as follows:
```python
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

> Note: the code snippet above is for running a model on a GPU - running on CPU takes a lot longer

You might run into an issue when you run the code snippet - `OSError: [Errno 28] No space left on device`. This is a result of not having enough space to download the model (in our case the model size is about 9.45GB) in the default directory stored as a variable called `HF_HOME`. Not to worry, the file path for this environment variable direct the downloads to a `.cache` folder in your home directory. Run `df -h` to see the available memory in the different file systems, so as to change the file path of `HF_HOME` to one that has more capacity. A way to do so is to run the following in a cell in your notebook:

```python
import os

os.environ['HF_HOME'] = '/run/cache'
```
Now you should be able to run the model code snippet without any errors. Now let's quickly dissect the code snippet. The main part really is the `tokenizer` - breaking down text into words, subwords, or characters into tokens and converting them into numerical IDs. This enables machine learning models to understand the inputs texts. Hence why your `input_ids` is a tensor array of token IDs.

![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/1c3e2fbe-dcb6-41ca-b9eb-de926892ab35)

### Rewriting the LLM Function

Keep in mind that we still use our `minsearch` package for searching through our database - in our case its the `FAQs` JSON file we had downloaded in the previous section.

There were three main functions encapsulated in the `rag` function that we had written before.
```python
def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer
```

But the only one that we are modifying is the `llm` function as we are changing the model from `GPT4` to `FLAN-T5-XL`.
```python
def llm(prompt, generate_params=None):
    if generate_params is None:
        generate_params = {}

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(
        input_ids,
        max_length=generate_params.get("max_length", 100),
        num_beams=generate_params.get("num_beams", 5),
        do_sample=generate_params.get("do_sample", False),
        temperature=generate_params.get("temperature", 1.0),
        top_k=generate_params.get("top_k", 50),
        top_p=generate_params.get("top_p", 0.95),
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
```
> Note: the snippet contains more parameters as we want to make the output longer.

Explanation of Parameters:

- `max_length`: Set this to a higher value if you want longer responses. For example, max_length=300.
- `num_beams`: Increasing this can lead to more thorough exploration of possible sequences. Typical values are between 5 and 10.
- `do_sample`: Set this to True to use sampling methods. This can produce more diverse responses.
- `temperature`: Lowering this value makes the model more confident and deterministic, while higher values increase diversity. Typical values range from 0.7 to 1.5.
- `top_k` and `top_p`: These parameters control nucleus sampling. `top_k` limits the sampling pool to the top `k` tokens, while `top_p` uses cumulative probability to cut off the sampling pool. Adjust these based on the desired level of randomness.


## 2.4 Phi 3 Mini

### Introduction to Microsoft's Phi3

The `PHI-3` (Phenomenal Holistic Intelligence 3) mini is a smalller version of the of the original `PHI-3` model also developed by Microsoft.

These models are built to be light and effective as compared to other open-sourced models in the market, and hence are dubbed as `Small Language Models` - a.k.a. `SML`.

The model is intended for commercial and research use in English. The model provides uses for applications which require:

1. Memory/compute constrained environments
2. Latency bound scenarios
3. Strong reasoning (especially code, math and logic)

###  Nvidia-AMI, Model Size and using Phi3

Run the `nvidia-smi` command on the terminal window in your `saturn cloud` instance to gather information on the following:
- GPU Utilisation
- Memory Usage
- Temperature
- Power Consumption
- Running processes on each GPU

![image](https://github.com/user-attachments/assets/b5046888-3435-4b95-b43b-4e569bb64226)

> Note: Interface comparison - this model has more of a chat style prompt rather that a completion style in the previous model

### Using Phi3 for RAG

More or less the structure is the same as the previous models - the only difference is the `llm function`:
```python
def llm(prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    
    return output[0]['generated_text'].strip()
```

You can find the Notebook: [huggingface-phi3-mini.ipynb](https://github.com/peterchettiar/LLMzoomcamp_2024/blob/main/Module-2-open-source-llm/huggingface-phi3-mini.ipynb) for more details.

## 2.5 Mistral-7B

### Introduction to Mistral-7B

`mistral-7b` is an open-sourced LLM that was developed by [Mistral AI](https://mistral.ai/). With 7 billion parameters, which is relatively small compared to other prominent LLMs, it demonstrates impressive performance as compared to other models like the `Llama 2 13B`. Perhaps the reason why it has gained significant attention in the AI community.

### Using Mistral-7B

Unlike the previous 2 models, `mistral-7b` is not so easily accessed. The is because `mistral-7b` is released under specific licence that requires users to agree to certain terms and conditions. So there couple of things we need to do:

1. Create an account on `huggingface` - do so if you have not done so already
2. Click on your profile icon at the top right-hand of page and select settings from the drop-down
3. On the left navigation panel, select `Access Tokens`
4. Click `create new token` on the top right-hand of page
5. You should see the following page - give a name for your token, I used `llm-course`, and click on `create token`
   ![image](https://github.com/user-attachments/assets/6f49ffd0-cb26-46fc-a86d-ed3954c457de)
6. Next, on your saturn cloud we need to create a new secret - same as an environment variable
7. As the below image shows, click on `Secrets` and then click on `New` to be able to add the name as well as the value of the token - the token being the one just created on `huggingface`
   ![image](https://github.com/user-attachments/assets/7ca9b0d7-143c-4ff7-b1b2-e39e6584aa48)
8. Once added, you should be able to see this on the `Secrets and Roles` section of your Jupyter Server as follows
   ![image](https://github.com/user-attachments/assets/0e8efcdf-33c7-4455-acf7-54d37a1e5737)

Now that we have saved `HF_TOKEN` in secrets, we can now call our model from `huggingface`. But before we do so, we need to login to perform the Hub Authentication. Follow the code snippet, this is pretty much the only difference as compared to the previously run notebooks.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

# Logging into HuggingFace

login(token=os.environ['HF_TOKEN'])

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", 
    device_map="auto",
    load_in_4bit = True
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
```
### Saving, Loading the model and the LLM function

The model is quite large, and so it is not advised to download the model for each run. We can download it once, and then subsequently save followed by loading it from a local directory. To do so, we can do the following for saving the `model` as well as `tokenizer`:

```python
# now let's save the model locally, so that we don't need to keep downloading them

model.save_pretrained("./mistral-7b-model")
tokenizer.save_pretrained("./mistral-7b-tokenizer")
```
For loading the model.

```python
# loading the model from local directory

model = AutoModelForCausalLM.from_pretrained("./mistral-7b-model")
tokenizer = AutoTokenizer.from_pretrained("./mistral-7b-tokenizer")
```

Last but not least, please find the following `llm` function for this model, and feel free to tweak the arguments to give it a more desired result.

```python
def llm(prompt):
    response = pipe(prompt, max_length=500, temperature=0.7, top_p=0.95, num_return_sequences=1)
    response_final = response[0]['generated_text']
    return response_final[len(prompt):].strip()
```
## 2.6 Exploring Open Source LLMs

We had just looked through a few models that are quite popular in the AI community, but there are a lot more open source models on HuggingFace. The question is **"How do we decide which model to choose?"**

The answer to the question is [open_llm_leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) on HuggingFace.The Open LLM Leaderboard is a platform that compares and ranks various open-source large language models based on their performance across different tasks,  including tasks related to common sense reasoning, reading comprehension, and mathematical abilities. Based on the tasks, you have an `Average` score which represents a composite measure of the model's performance across all evaluated tasks. Your choice of model can be based on either the overall score or based on the performance against the task. There are multiple tasks available on the leaderboard, but here's a few that includes their respective descriptions as well:

1. `IFEval` - IFEval is a dataset designed to test a model’s ability to follow explicit instructions, such as “include keyword x” or “use format y.” The focus is on the model’s adherence to formatting instructions rather than the content generated, allowing for the use of strict and rigorous metrics.
2. `BBH` - Big Bench Hard (BBH) is a subset of 23 challenging tasks from the BigBench dataset to evaluate language models. The tasks use objective metrics, are highly difficult, and have sufficient sample sizes for statistical significance. They include multistep arithmetic, algorithmic reasoning (e.g., boolean expressions, SVG shapes), language understanding (e.g., sarcasm detection, name disambiguation), and world knowledge. BBH performance correlates well with human preferences, providing valuable insights into model capabilities.
3. `MATH` - MATH is a compilation of high-school level competition problems gathered from several sources, formatted consistently using Latex for equations and Asymptote for figures. Generations must fit a very specific output format.
4. `GPQA` - Graduate-Level Google-Proof Q&A Benchmark is a highly challenging knowledge dataset with questions crafted by PhD-level domain experts in fields like biology, physics, and chemistry. These questions are designed to be difficult for laypersons but relatively easy for experts. The dataset has undergone multiple rounds of validation to ensure both difficulty and factual accuracy.
5. `MuSR` - Multistep Soft Reasoning is a new dataset consisting of algorithmically generated complex problems, each around 1,000 words in length. The problems include murder mysteries, object placement questions, and team allocation optimizations. Solving these problems requires models to integrate reasoning with long-range context parsing. Few models achieve better than random performance on this dataset.
6. `MMLU-PRO` - Massisve Multitask Language Understanding - Professional is a refined version of the MMLU dataset, which has been a standard for multiple-choice knowledge assessment. Recent research identified issues with the original MMLU, such as noisy data (some unanswerable questions) and decreasing difficulty due to advances in model capabilities and increased data contamination. MMLU-Pro addresses these issues by presenting models with 10 choices instead of 4, requiring reasoning on more questions, and undergoing expert review to reduce noise. As a result, MMLU-Pro is of higher quality and currently more challenging than the original.

## 2.7 Running LLMa locally without a GPU using Ollama

### Setting up and running Ollama

Ollama is an open-source tool that allows users to run large language models (LLMs) locally on their personal computers. It simplifies the process of downloading, setting up, and interacting with various AI models like Llama 2, Mistral, and others.
Key features:

1. Easy installation and use
2. Runs models locally for privacy and offline access
3. Supports multiple open-source LLMs
4. Simple command-line interface
5. Ability to customize and create model configurations
6. Includes an API for integration with other applications

Ollama is popular among developers and AI enthusiasts who want to experiment with LLMs without relying on cloud services. It's designed to be efficient, allowing even larger models to run on standard consumer hardware.

Setting up is quite easy, I am using `codespaces` so I ran the following command for linux based machines to install `ollama`:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Next, I run the terminal command `ollama start` to initialise and start the `ollama` service on our machine. Now that we have `ollama` running, we can run the command `ollama run phi3` to be able to interact with the model locally. It's so easy to use!

That would be the simplest way to use it, but there are two other methods to implement `ollama`:
1. Replacement for `OpenAI` API
2. Running `ollama` in docker 

### Replacement for OpenAI API

Interesting point to be shared in this section is that the `ollama` API is designed to be compatible with the `openai` API format.

What this means is that we are able to use the `openai` Python Library, but its being configured to work with `ollama`. And all we have to do is to configure our connection to the `openai` API as follows:

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)
```

Here's what's happening:

1. `OpenAI Client`: The code is importing and using the OpenAI client library.
2. `Custom Base URL`: Instead of using OpenAI's servers, it's setting the `base_url` to `'http://localhost:11434/v1/'`, which is the local API endpoint for `ollama`.
3. API Key: `The api_key` is set to 'ollama', which is a placeholder value. `ollama` doesn't actually require an API key for local use.

This setup allows you to use the `OpenAI` Python library's interface to interact with Ollama-served models running locally on your machine. It's a clever way to use a familiar API structure (OpenAI's) with locally hosted models through Ollama.

Please note that the `llm` function should be amended as well, instead of `gpt-4o`, the model argument should be specified to one of the `ollama` models. In our example, we use `phi3`.

```python
def llm(prompt):
    response = client.chat.completions.create(
        model='phi3',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

Please find the implementation in a jupyter notebook [here](https://github.com/peterchettiar/LLMzoomcamp_2024/blob/main/Module-2-open-source-llm/ollama_example.ipynb).

> Note: In order to run the notebook, please make sure that ollama is installed and running. Also, do check if the model you want to use is pulled first before running.

### Running Ollama in Docker

Another interesting way of running `ollama` is using `docker`. Some people prefer using docker as it is self-contained although `ollama` only has one executable.

command for running `ollama` container on `docker`.
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
Lets break down the command:
`docker run` : start a new container from image, if image is not available it would be pulled from `dockerhub`
`-d` : run container as daemon or detached mode, meaning it runs in the background
`-v ollama:/root/.ollama` : This creates a volume named 'ollama' and mounts it to `/root/.ollama` in the container. This is used for persistent storage of Ollama data
`-p 11434:11434` : This maps port 11434 on your host machine to port 11434 in the container. This is the default port Ollama uses for its API.
`--name ollama` : This assigns the name 'ollama' to the running container for easy reference.
`ollama/ollama` : This specifies the Docker image to use, in this case, the official Ollama image

Now, that we have the `ollama` server running, we want to use the `client` to be able to ask questions.

The command for doing so is (i.e. running the command inside the container):
```bash
docker exec -it ollama ollama run phi3
```
`docker exec` : runnning a command inside a container
`it` : run the command in interactive terminal
`ollama` : name of the container
`ollama run phi3` : command to be run on the host - in this case we want to pull model as well as run it

### Customize a model

Import from GGUF

Ollama supports importing GGUF models in the MOdelfile:

1. Create a file named `Modelfile`, with a `FROM` instruction with the local filepath to the model you want to import.

```bash
FROM ./vicuna-33b.Q4_0.gguf
```

2. Create the model in Ollama
```bash
ollama create example -f Modelfile
```

3. Run the model
```bash
ollama run example
```

## 2.8 Ollama & Phi3 + Elastic in Docker-Compose

In this section, we will be building on top of the previous section where we had run `ollama` locally, ran it as a replacement for `openai` API as well as running it from a docker container. But that was very specific to only running `ollama` LLM models. However, recall that in our RAG architecture we had another component other than the LLM model, our knowledge base. 

So far for the purpose of illustration we had implemented the toy search engine using `minsearch.py` library for ease of running it in `saturn cloud`. And we had mentioned in the previous module that this was to be replaced by `elasticsearch` - a powerful open-source search and analytics engine, as this makes the architecture more production ready.

And the best approach for running the models together is using docker. So let's go through in the next section on how to create a docker compose `YAML` file, and the command to run multiple container as specified in the `YAML` file.

### Creating a docker compose file

A docker-compose `YAML` file is nothing more than a `requirements.txt` file when setting up a virtual environment for a new project. But instead of installing dependencies, you are defining the services of the container applications you would like to spin up. In our case, the two services are `elasticsearch` and `ollama`.

And defining our services is essentially the reformatting of docker commands that we had run previously. Let's take the `ollama` docker commands as an example. We had the following 2 commands:
```bash
1. docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
2. docker exec -it ollama ollama run phi3
```

These commands are to firstly spin up and run the `ollama` container from its docker image, followed by runnning the `phi3` model from inside the container. So if I had to convert to a docker-compose file, the same command would look like the following in the `YAML` file:
```yaml
version: '3.8'

services:
    ollama:
        image: ollama/ollama
        container_name: ollama
        volumes:
            - ollama:/root/.ollama
        ports:
            - "11434:11434"
        deploy:
            resources:
                limits:
                    memory: 6G

volumes:
  ollama:
    name: ollama
```

Notice that they are almost the same. `init: true` ensures that the container uses an init process, which can help with process management and zombie process reaping. And `command` is to specify to start the `ollama` server in the background, wait for 10 seconds before running the `phi3` model. 

> Note: A new container needs to be created using this docker-compose file with the command tweaked according to the model that we want to use.

Similary, if we were to convert the command that we used to run `elasticsearch` with docker:
```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

It would look something like this:
```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.4.3
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
      - "9300:9300"
```

So combining both together would give you something like thie [docker-compose.yaml](https://github.com/peterchettiar/LLMzoomcamp_2024/blob/main/Module-2-open-source-llm/docker-compose.yaml) file. 

### Modifying module 1 notebook


### RAG flow functions and response from Ollama