| Title | Link | Q | Code | Summary |
| :--- | :--- | ---: | :--- | :--- |
| Neural Ordinary Differential Equations | [Paper](https://arxiv.org/pdf/1806.07366) | 1 |  - | Conceptual paper. It also contains general idea of latent dynamics for time series on which we can build regression or **classification** |
| Neural ODE Control for Classification, Approximation, and Transport | [Paper](https://arxiv.org/pdf/2104.05278) | 1 | - | Let's assume we have 1-layer NN as field function and $N$ points to classify to $M$ classes. Classes can be any partion of $\mathbb{R}^d$. They set up problem as finding such control (i.e. 1-layer NN) such that each point after time $T$ flows to correspodning class. |
| Neural Ordinary Differential Equations for Hyperspectral Image Classification | [Paper](https://www2.umbc.edu/rssipl/people/aplaza/Papers/Journals/2020.TGRS.ODE.pdf) | 1 | - | Using general latent NODE for classifying complex scenes collated from HSI |
| Structural identification with physics-informed neural ordinary differential equations | [Paper](https://www.sciencedirect.com/science/article/pii/S0022460X21002686?casa_token=7Sgwj7Cz0PAAAAAA:O7chyfrsZ0QSOwlZZ3r0uEwzv9bks7ME6hO_jMXuZDvvuXsASYYNOjOWGf6dODeNOSHQc52Weq4) | ? | - | Use NODE in linkage with physics-informed terms in the dynamics to model structural dynamics |
| Neural modal ordinary differential equations: Integrating physics-based modeling with neural ordinary differential equations for modeling high-dimensional monitored structures | [Paper](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/FAF0447B372F8EDFE4DA13DC350ADAD9/S2632673622000351a.pdf/neural-modal-ordinary-differential-equations-integrating-physics-based-modeling-with-neural-ordinary-differential-equations-for-modeling-high-dimensional-monitored-structures.pdf) | ? | - | Pretty much the same |
| A Neural Network Estimation of Ankle Torques From Electromyography and Accelerometry | [Link](https://ieeexplore.ieee.org/abstract/document/9513274) | ? | - | Extract features from time series using windowing and physical premises. Use different models including NODE to regress ankle torques from measurements. Says that LSTM is the best and use it for forecasting. |
| Detecting Strange Attractors in Turbulence | :--- | ---: | :--- | Takens theorem |
| SSA-based approaches to analysis and forecast of multidimensional time series | :--- | ---: | :--- | Golyndina SSA |
| Physically informed neural networks for the simulation and data-assimilation of geophysical dynamics | [Link](https://sci-hub.ru/10.1109/igarss39084.2020.9323081) | Conferences and Proceedings| :--- | Guys (partially) using Takens theorem to restore trajectories and using NODE (though do not name it like this) to learn dynamics + proposes regulating terms for stability (derived from Lyapunov). They do not exactly use Takens but augmented latent space. |
| Learning Latent Dynamics for Partially-Observed Chaotic Systems | [Paper](https://openreview.net/pdf?id=BygMreSYPB) | Conferences and Proceedings | - | Guys doing literally the same thing! And the refernce a lot of neural ode approaches and difference between takens approach and theirs. |
| Tangent Space Causal Inference: Leveraging Vector Fields for Causal Discovery in Dynamical Systems | [Paper](https://arxiv.org/pdf/2410.23499) | - | - | Guys proposed alternative for CCM in tangent space. They used Takens theorem as a mean. They also used NODE in one of their experiments. |




Моя работа о подходах в классификации временных рядов, используя модель скрытых динамических систем и NODE. Статистический, классификация в пространстве модели, байесовский подходы. Нужно рассказать о достоинствах и недостатках каждого. Графовое представление.

В интро дать референс на общий подход NODE к временным рядам из оригинальной статьи + примеры применения этого. Указать на его недостаток с $z_0$ и др. Указать на интересный способ из второй статьи. Указать, что у нас классификация в любом случае через регрессию.

Бывают разные работы: можно ислледовать какие-нибудь теоретические свойства модели/алгоритма (сходимость, существование, единственность ...). Можно предложить численный алгоритм, основываясь на теоретической модели. Моя работа о "новых" методах классификации: я предлагаю модель данных и численный алгоритм классификации.

