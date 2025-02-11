# Machine Learning

### Alexander Del Toro Barba, PhD

[www.deltorobarba.com](https://www.deltorobarba.com) $\cdot$ [Google Scholar](https://scholar.google.com/citations?hl=en&user=fddyK-wAAAAJ)

<br>

<img src="https://raw.githubusercontent.com/deltorobarba/repo/master/sciences_4000.png" alt="sciences">

<br>

* https://jax-ml.github.io/scaling-book/
* https://github.com/GoogleCloudPlatform/generative-ai
* https://cloud.google.com/transform/top-five-gen-ai-tuning-use-cases-gemini-hundreds-of-orgs?e=48754805&hl=en
* https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/
* https://www.arxiv.org/abs/2501.18708

## Infrastructure AI
* [Ray on Vertex](https://github.com/deltorobarba/machinelearning/blob/main/ray.ipynb) with PyTorch for multi-node and multi-GPU training
* [Nvidia](https://github.com/deltorobarba/machinelearning/blob/main/nvidia.ipynb) - Accelerators and Architectures
* [vLLM for GPUs](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-gemma-gpu-vllm) with Gemma
* [Google Hypercompute recipes](https://github.com/AI-Hypercomputer/gpu-recipes) Reproducible benchmark recipes for GPUs (Github)
* [A3 supercomputers with NVIDIA H100 GPUs](https://cloud.google.com/blog/products/compute/introducing-a3-supercomputers-with-nvidia-h100-gpus?e=48754805)
* [Running HuggingFace Llama Training on Cloud TPUs](https://github.com/pytorch-tpu/transformers/blob/alanwaketan/flash_attention/USER_GUIDE.md)
* [Reproducibility](https://github.com/gclouduniverse/reproducibility) of specific workloads of training and serving
* [HuggingFace Llama 2 7B Training on Cloud TPUs](https://github.com/pytorch-tpu/transformers/blob/alanwaketan/flash_attention/USER_GUIDE.md) 

## Generative AI - Tasks
* [Chaining](https://github.com/deltorobarba/machinelearning/blob/main/chaining.ipynb) 
* [Caching](https://github.com/deltorobarba/machinelearning/blob/main/caching.ipynb) 
* [Grounding](https://github.com/deltorobarba/machinelearning/blob/main/grounding.ipynb)
* [Finetuning](https://github.com/deltorobarba/machinelearning/blob/main/finetuning.ipynb)
* [Reasoning](https://github.com/deltorobarba/machinelearning/blob/main/reasoning.ipynb)
* [Seed](https://github.com/deltorobarba/machinelearning/blob/main/seed.ipynb) 
* [Evaluation](https://github.com/deltorobarba/machinelearning/blob/main/evaluation.ipynb) 
* [Monitoring](https://github.com/deltorobarba/machinelearning/blob/main/monitoring.ipynb)
* [LLM Journey](https://github.com/deltorobarba/machinelearning/blob/main/slides.pdf)


## Generative AI - Models
* [Gemini 2.0](https://github.com/deltorobarba/machinelearning/blob/main/gemini2.ipynb) 
* [Anthropic](https://github.com/deltorobarba/machinelearning/blob/main/anthropic.ipynb) 
* [Gemma](https://github.com/deltorobarba/machinelearning/blob/main/gemma.ipynb) 
* [Imagen](https://github.com/deltorobarba/machinelearning/blob/main/imagen.ipynb) 
* [Llama](https://github.com/deltorobarba/machinelearning/blob/main/llama.ipynb) 
* [Diffusion](https://github.com/deltorobarba/machinelearning/blob/main/llama.ipynb) 

## Predictive AI
* [BigQuery](https://github.com/deltorobarba/machinelearning/blob/main/bigquery.ipynb) IoT and energy forecasting
* [OCR](https://github.com/deltorobarba/machinelearning/blob/main/ocr.ipynb) Text recognition
* [Pipeline](https://github.com/deltorobarba/machinelearning/blob/main/pipeline.ipynb) MLOPs AutoML tabular regression pipelines

## Google
* [GenOps: the evolution of MLOps for gen AI](https://cloud.google.com/blog/products/ai-machine-learning/learn-how-to-build-and-scale-generative-ai-solutions-with-genops?hl=en&e=48754805) blog post
* [Google Codelabs](https://codelabs.developers.google.com/?category=aiandmachinelearning) developer repo guides
* [Google Gemini](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/sample-apps/e2e-gen-ai-app-starter-pack) - End-to-End Gen AI App Starter Pack
* [Google AI Studio](https://aistudio.google.com/prompts/new_chat)
* [NotebookLM](https://notebooklm.google/)
* [illuminate](https://illuminate.google.com/home) - Research Audio 

## Third-Party
* [Building A Generative AI Platform](https://huyenchip.com/2024/07/25/genai-platform.html) by Chip Huyen
* [Predibase](https://docs.predibase.com/user-guide/fine-tuning/turbo_lora#how-to-train-with-lora) LoRA Tuning on top of custom weights based on Llama 3.x model/architecture
* [replicate](https://replicate.com/) - platform to fine-tuning models
* [unsloth](https://github.com/unslothai/unsloth) fine-tuning of LLMs & Vision LMs 2.2x faster and use 80% less VRAM
  * [QLoRA](https://medium.com/@dillipprasad60/qlora-explained-a-deep-dive-into-parametric-efficient-fine-tuning-in-large-language-models-llms-c1a4794b1766)
  * [PEFT, QLORA with unsloth](https://medium.com/@tejpal.abhyuday/optimizing-language-model-fine-tuning-with-peft-qlora-integration-and-training-time-reduction-04df39dca72b)
* [kserve](https://kserve.github.io/website/latest/) Inference Platform on Kubernetes, built for highly scalable use cases
* [triton](https://developer.nvidia.com/triton-inference-server) (Nvidia) designed to maximize performance of inference on GPUs and CPUs
* [ollama](https://hub.docker.com/r/ollama/ollama) - Deploy small language models locally
* [gradio](https://www.gradio.app/) - Fastest way to demo machine learning model with friendly web interface
* [anyscale](https://www.anyscale.com/) - AI framework to fully utilize every GPU with RayTurbo
* [firebase](https://firebase.google.com/docs/genkit#basic-generation)
* [LlamaIndex](https://docs.llamaindex.ai/en/stable/) Indexing and querying data. Features like PageWise indexing and auto-merging ensure accurate retrieval of relevant information.
* www.3blue1brown.com - Visualization of machine learning
