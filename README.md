# deepCloak

deepCloak is a framework that came about with the idea that;
\A. side-channel leakage is cumbersome, costly and time consuming to process therefore it is very likely that attackers are adopting AI techniques to automate the process.
\B. Deep learning models can be fooled via carefully crafted adversarial samples. If attackers are also using deep learning models, we can turn this inherent weekness into a defensive tool.

So, we created this framework, deepCloak that takes side-channel leakage data of a process as input and calculates the necessary adversarial perturbations to morph the side-channel trace into an unrelated process, essentially cloaking it against side-channel listeners. We use the publicly available FoolBox (add citation) library to craft adversarial samples.

Moreover, we investigate defenses against adversarial samples and show that even in the presence of adversarial re-training and defensive distillation, models can still be 'fooled'.
