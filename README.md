# BPL_TEST2_Perfusion

Basic example of perfusion cultivation using an FMU from Bioprocess Library *for* Modelica. 
In this notebook you see several examples of how to interact with the FMU to provide process insight.
Below script and diagram with a typical simulations of perfusion cultivation that you will get at the end of the notebook.

![](Fig2_BPL_TEST2_Perfusion_steps.png)

You start up the notebook in Colab by pressing here
[start BPL notebook](https://colab.research.google.com/github/janpeter19/BPL_TEST2_Perfusion/blob/main/BPL_TEST2_Perfusion.ipynb).
Then you in the menu choose Runtime/Run all.

The installation takes just a few minutes. The subsequent execution of the simulations of microbial growth take just a second or so. You can continue in the notebook and make new simulations and follow the examples given.

Note that:
* The script occassionaly get stuck during installation. Then just close the notebook and start from scratch.
* Runtime warnings are at the moment silenced. The main reason is that we run with an older combination of PyFMI and Python that bring depracation warnings of little interest. 
* Remember, you need to have a google-account!

Just to be clear, no installation is done at your local computer.

License information:
* The binary-file with extension FMU is shared under the permissive MIT-license
* The other files are shared under the GPL 3.0 license
