# Week 2 notes

### Table of contents

- [2.1 Open-Source LLMs - Introduction](#21-open-source-llms---introduction)
    - [Open-Source LLMs](#open-source-llms)
    - [Replacing the LLM box in the RAG flow](#replacing-the-llm-box-in-the-rag-flow)
- [2.2 Using a GPU in Saturn Cloud](#22-using-a-gpu-in-saturn-cloud)
    - [Registering in Saturn Cloud](#registering-in-saturn-cloud)
    - [Configuring Secrets and Git](#configuring-secrets-and-git)
    - [Creating an instance with GPU](#creating-an-instance-with-gpu)

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
8. From the git repositories page, select the `New` button at the top right corner. Here, you can add a repository via remote URL (this is the link you’d use when running `git clone` in the terminal).
![image](https://github.com/peterchettiar/LLMzoomcamp_2024/assets/89821181/735704c6-9377-46fe-8356-b90aeea31d67)
9. If you still feel a little lost, please read this [documentation](https://saturncloud.io/docs/using-saturn-cloud/gitrepo/#:~:text=Click%20on%20Create%20an%20SSH,the%20SSH%20key%20creation%20form.&text=After%20you%20have%20a%20key,to%20create%20the%20secure%20connection.)

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
