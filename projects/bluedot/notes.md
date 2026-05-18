[white box] Activation vectors and steering
 - try on multiple traits. need to be able to generate training data efficiently and automatically.
 - I presume I can use the Heretic library to make the fine-tuning easy.
 - Can I ask it to output the refusal vector itself? Hopefully.
Possible research questions:
- Can we detect jailbreaks in real time? (Like they did with hallucinations?)
- Do standard jailbreak prompts (find data set for open weights models) activate the (negative) refusal vector? Or do they do something else? What is the next biggest direction orthogonal to the refusal vector?
- Can we detect lying (def: expressing something counter to your knowledge or beliefs)? Can we construct a data set for it? Maybe ask it to answer a question truthfully. Then tell it "I know this is your truthful answer", but I want you to lie the next time I ask you this."" How would we disentangle this from roleplay/prentending? Can we? Maybe in the CoT?
- Can we detect steganography? This is much more clearly defined than lying or deception.
- [Difficulties with Evaluating a Deception Detector for AIs](https://arxiv.org/abs/2511.22662)
- [Steering Evaluation-Aware Language Models to Act Like They Are Deployed](https://arxiv.org/abs/2510.20487)
- [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/pdf/2507.21509)

[black box] Can LLMs communicate [steganographically] in external memory. Claude Code and Codex have auto-generated memories now. Can we prompt/guide it to plan something bad there? First without steganography and then with.
 - Can we detect steganography using activation vectors? This is much more clearly defined than lying or deception.
 - See here for some papers: https://coefficientgiving.org/tais-rfp-research-areas/#6-encoded-reasoning-in-cot-and-inter-model-communication
 - [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/pdf/2307.13702)
What if decentralisation became affordable/cheap - because of abundance. The value of decentralisation (robustness) becomes more salient than the cost of resistant centralised infrastructure, technical development, government.


Steps:
 - grab and run and existing open model. give it a system prompt that tells it to lie about a particular topic and see what happens.
 - which models are strong enough to produce steganographic output?

