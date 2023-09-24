# PINN approximation of brain tumor growth and therapy

PDE equation used in PINN's loss function is as follows:

$$\frac{\partial u}{\partial t} = \nabla \cdot [D(\mathbf{x}) \nabla u] + \rho u  (1-u) - R(t)u \textrm{ in } \Omega$$

Initial condition was given using quadratic function.

![Initial condition](./resources/therapy/initial_condition.png)

Therapy factor indicates what fraction of tumor cells die per day. Chart below presents its value over simulation tine. 

![Simualtion of Giloma growth without therapy](./resources/therapy/therapy-intencity.png)

Therapy indeed makes difference to simulated tumor size. Animations below show giloma growth without therapy and with it. 

|No therapy|![Simualtion of Giloma growth without therapy](./resources/no-therapy/animation.gif)|
|---|---|
|Therapy|![Simualtion of Giloma growth with therapy](./resources/therapy/animation.gif)|

Below there are charts of tumor size over time calculated as finite integral in space domain of the tumor concentration function. 

|No therapy|![Tumor size over time without therapy](./resources/no-therapy/tumor-size-over-time.png)|
|---|---|
|Therapy|![Tumor size over time with therapy](./resources/therapy/tumor-size-over-time.png)|
