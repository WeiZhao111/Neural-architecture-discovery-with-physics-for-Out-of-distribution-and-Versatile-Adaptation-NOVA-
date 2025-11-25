import numpy as np
import matplotlib.pyplot as plt
from smt.applications.mfk import MFK, NestedLHS

# low fidelity model
def lf_function(x):
    import numpy as np

    return (
        0.5 * ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)
        + (x - 0.5) * 10.0
        - 5
    )

# high fidelity model
def hf_function(x):
    import numpy as np

    return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)

# Problem set up
xlimits = np.array([[0.0, 1.0]])
xdoes = NestedLHS(nlevel=2, xlimits=xlimits, random_state=0)
xt_c, xt_e = xdoes(7)

# Evaluate the HF and LF functions
yt_e = hf_function(xt_e)
yt_c = lf_function(xt_c)

# sm = MFK(theta0=xt_e.shape[1] * [1.0])
sm = MFK(n_start = 50)


# low-fidelity dataset names being integers from 0 to level-1
sm.set_training_values(xt_c, yt_c, name=0)
# high-fidelity dataset without name
sm.set_training_values(xt_e, yt_e)

# train the model
sm.train()

x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)

# query the outputs
y = sm.predict_values(x)
mse = sm.predict_variances(x)
derivs = sm.predict_derivatives(x, kx=0)

plt.figure()

plt.plot(x, hf_function(x), label="reference")
plt.plot(x, y, linestyle="-.", label="mean_gp")
plt.scatter(xt_e, yt_e, marker="o", color="k", label="HF doe")
plt.scatter(xt_c, yt_c, marker="*", color="g", label="LF doe")

plt.legend(loc=0)
plt.ylim(-10, 17)
plt.xlim(-0.1, 1.1)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.show()
