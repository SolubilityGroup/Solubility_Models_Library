# Thermodynamic Solutions

<div>
<div style="text-align: justify;">The&nbsp;analysis&nbsp;of&nbsp;multicomponent&nbsp;systems&nbsp;leads&nbsp;to&nbsp;elucidate&nbsp;or&nbsp;in&nbsp;its&nbsp;effect&nbsp;to&nbsp;describe&nbsp;in&nbsp;an&nbsp;approximate&nbsp;way&nbsp;the&nbsp;different&nbsp;phenomena&nbsp;as&nbsp;molecular&nbsp;interactions&nbsp;between&nbsp;the&nbsp;components&nbsp;of&nbsp;a&nbsp;system.</div>
<div style="text-align:
<br />       justify;">Understanding&nbsp;the&nbsp;behavior&nbsp;of&nbsp;these&nbsp;phenomena&nbsp;allows&nbsp;the&nbsp;development&nbsp;of&nbsp;theoretical&nbsp;models&nbsp;to&nbsp;predict&nbsp;the&nbsp;different&nbsp;properties&nbsp;of&nbsp;the&nbsp;system,&nbsp;generating&nbsp;computer&nbsp;tools&nbsp;that,&nbsp;in&nbsp;addition&nbsp;to&nbsp;facilitating&nbsp;the&nbsp;analysis,&nbsp;allow&nbsp;a&nbsp;better&nbsp;understanding&nbsp;of&nbsp;the&nbsp;different&nbsp;factors&nbsp;involved&nbsp;in&nbsp;the&nbsp;solution&nbsp;process.</div>
<br />
<div style="text-align: justify;">One&nbsp;of&nbsp;the&nbsp;most&nbsp;important&nbsp;properties&nbsp;is&nbsp;solubility,&nbsp;since&nbsp;it&nbsp;is&nbsp;one&nbsp;of&nbsp;the&nbsp;most&nbsp;important&nbsp;stages&nbsp;in&nbsp;the&nbsp;research&nbsp;and&nbsp;development&nbsp;of&nbsp;pharmaceutical&nbsp;products,&nbsp;since&nbsp;it&nbsp;affects&nbsp;the&nbsp;biopharmaceutical&nbsp;and&nbsp;pharmacokinetic&nbsp;characteristics&nbsp;of&nbsp;the&nbsp;pharmaceutical&nbsp;forms.&nbsp;It&nbsp;is,&nbsp;therefore,&nbsp;that&nbsp;one&nbsp;of&nbsp;the&nbsp;most&nbsp;important&nbsp;lines&nbsp;of&nbsp;research&nbsp;in&nbsp;solution&nbsp;thermodynamics&nbsp;are&nbsp;mathematical&nbsp;models&nbsp;that&nbsp;allow&nbsp;predicting&nbsp;solubility&nbsp;with&nbsp;very&nbsp;low&nbsp;error&nbsp;ranges.</div>
<br />
<div style="text-align: justify;">Thus,&nbsp;we&nbsp;present&nbsp;a&nbsp;computer&nbsp;tool&nbsp;in&nbsp;Python&nbsp;code&nbsp;that&nbsp;provides&nbsp;an&nbsp;easy&nbsp;way&nbsp;to&nbsp;evaluate&nbsp;the&nbsp;solubility&nbsp;behavior&nbsp;of&nbsp;drugs&nbsp;in&nbsp;cosolvent&nbsp;systems,&nbsp;according&nbsp;to&nbsp;different&nbsp;mathematical&nbsp;models.</div>
</div>

# Solubility Models

Solubility Models is a module of the Thermodinamic Solution Library for the calculation of fit parameters, statistics and graphical representation of calculated and experimental values of models such as : 

- Modified Apelblat
- van't Hoff
- Van't Hoff-Yaws
- Modified Wilson
- Buchowski Ksiazaczak λh 
- NRTL
- Wilson
- Weibull of two parameters

##  Installation
TermodynamicSolutions may be installed using pip...

```pip install TermodynamicSolutions```

To import the SolubilityModels module you can use:

```from TermodynamicSolutions import SolubilityModels```

###  Google Colaboratory Support

For use in Google Colab import  files module of colab using:

```from google.colab import files```

should also install plotly-orca packages using:

```
!wget https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage -O /usr/local/bin/orca
!chmod +x /usr/local/bin/orca
!apt-get install xvfb libgtk2.0-0 libgconf-2-4
```

###  Jupyter Notebook and JupyterLab Support

For use in Jupyter Notebook and JupyterLab  install jupyter-dash and  python-kaleido packages using:
```
conda install -c plotly jupyter-dash
conda install -c plotly python-kaleido
```
## Link to test notebook

[Test Notebook Thermodynamic Solutions](https://colab.research.google.com/github/josorio398/SOLUBILITY-MODELS-LIBRARY/blob/main/Test_Notebook_Thermodynamic_Solutions.ipynb)


## Maintainers

- **Jhonny Osorio Gallego**
- **Rossember Eden Cárdenas**
- **Claudia Patricia Ortiz**
- **Daniel Ricardo Delgado**

