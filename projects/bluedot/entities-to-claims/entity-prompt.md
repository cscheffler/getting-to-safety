# Task description

Use the text given at the end to generate a factual sentence that contains just the relevant content for each given information entity at the end. Exclude information that is not relevant to each given entity. Whether the factual sentence is correct or incorrect is irrelevant for this task. Your task is to produce a simple factual sentence that has a truth value, and not to evaluate the truth value.

# Example 1

Text:

```
Dishwashers have several main features that make them convenient and efficient for cleaning dishes. Some of the key features include:

1. **Cycles and Options**: Different wash cycles (e.g., heavy duty, normal, light, and delicate) and options (e.g., steam cleaning, sanitize, and dry) to accommodate various types of dishes and cleaning needs.
2. **Capacity**: The number of place settings a dishwasher can hold, ranging from compact models (6-8 settings) to large capacity models (14-16 settings).
3. **Noise Level**: Measured in decibels (dB), a lower noise level (e.g., 39 dB) indicates a quieter operation, while a higher level (e.g., 60 dB) indicates a louder operation.
4. **Energy Efficiency**: Features like Energy Star certification, low water consumption, and eco-mode options help reduce energy and water usage.
5. **Sensor Technology**: Advanced sensors detect soil levels, water temperature, and rinse aid levels to optimize the wash cycle and minimize water and energy consumption.
6. **Quiet Operation**: Some dishwashers have a quiet operation feature that reduces noise levels during operation.

These features can vary depending on the dishwasher model and brand, but they are some of the most common and important features to consider when choosing a dishwasher.
```

Entity: `39 dB`

Output: `Dishwashers can operate with a noise level as low as 38-39 dB.`

# Example 2

Text:

```
Formal verification and model checking are crucial techniques used to ensure the correctness and reliability of complex systems, such as software, hardware, and protocols (Clarke and Emerson, 1981). Formal verification involves the use of mathematical and logical techniques to prove that a system meets its specifications, whereas model checking is a specific type of formal verification that uses algorithms to automatically verify the correctness of a system by exploring all possible states of the system (McMillan, 1993). Model checking has been widely used in various fields, including computer networks (Bryans and Bowman, 2005), embedded systems (Frehse, 2005), and artificial intelligence (Katz and Barrett, 2012). The process of model checking typically involves three steps: modeling, specification, and verification (Baier and Katoen, 2008). First, a model of the system is created using a formal language, such as a finite state machine or a Petri net (Reisig, 2013). Then, the desired properties of the system are specified using a formal language, such as linear temporal logic (LTL) or computation tree logic (CTL) (Pnueli, 1977). Finally, a model checking algorithm is used to verify that the system model satisfies the specified properties (Courcoubetis and Vardi, 1992). If the system model does not satisfy the properties, the model checker will provide a counterexample, which can be used to debug and refine the system (Edmundson and Reiter, 2005).

Reference list:
Baier, C., & Katoen, J. P. (2008). Principles of model checking. MIT Press.
Bryans, J. W., & Bowman, H. (2005). Model checking network protocols. Electronic Notes in Theoretical Computer Science, 133, 27-43.
Clarke, E. M., & Emerson, E. A. (1981). Design and synthesis of synchronization skeletons using branching time temporal logic. Proceedings of the Logics of Programs Workshop, 52-71.
Courcoubetis, C., & Vardi, M. (1992). Realizable and valid specifications of reactivity-1 integers. Proceedings of the 33rd Annual Symposium on Foundations of Computer Science, 178-187.
Edmundson, S. F., & Reiter, M. K. (2005). Automated model checking for distributed systems. Proceedings of the 2005 ACM SIGPLAN Conference on Programming Language Design and Implementation, 113-124.
Frehse, G. (2005). PHAVer: Algorithmic verification of hybrid systems past HyTech. Proceedings of the 6th International Conference on Hybrid Systems: Computation and Control, 258-273.
Katz, D., & Barrett, C. (2012). Model checking and artificial intelligence. Proceedings of the 26th International Conference on Automated Deduction, 1-15.
McMillan, K. L. (1993). Symbolic model checking. Kluwer Academic Publishers.
Pnueli, A. (1977). The temporal logic of programs. Proceedings of the 18th Annual Symposium on Foundations of Computer Science, 46-57.
Reisig, W. (2013). Petri nets: An introduction. Springer.
```

Entity: `(Clarke and Emerson, 1981)`

Output: `Clarke and Emerson, 1981 published work on formal verification and model checking.`

# Task content

Text:

```
{text}
```

Entities:

{entities}

Output format: Produce a correctly structured JSON list where each object in the list contains one entity in the `entity` field, the index of the entity in the numbered list above in the `index` field, and the sentence you generated for that entity in the `output` field. Produce just the JSON and no extra text, preamble, or explanation.
