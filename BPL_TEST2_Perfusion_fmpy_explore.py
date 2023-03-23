# Figure - Simulation of perfusion reactor
#          with a set of functions and global variables added to facilitate explorative simulation work.
#          The general part of this code is called FMU-explore and is planned to be avaialbe as a separate package.
#
# GNU General Public License v3.0
# Copyright (c) 2022, Jan Peter Axelsson, All rights reserved.
#------------------------------------------------------------------------------------------------------------------
# 2022-06-01 - Introduced as default a simpler plot and kept the old more complete plot as well
# 2022-10-17 - Updated for FMU-explore 0.9.5 with disp() that do not include extra parameters with parLocation
# 2023-02-08 - Updated to FMU-explore 0.9.6e
# 2023-02-13 - Consolidate FMU-explore to 0.9.6 and means parCheck and par() udpate and simu() with opts as arg
# 2023-02-28 - Update FMU-explore for FMPy 0.9.6 in one leap and added list key_variables for logging
# 2023-03-22 - Update FMU-explore for FMPy 0.9.7b and ensured all states logged by using key_variables for now
# 2023-03-23 - Update FMU-explore 0.9.7c
#------------------------------------------------------------------------------------------------------------------

# Setup framework
import sys
import platform
import locale
import numpy as np 
import matplotlib.pyplot as plt 

from fmpy import simulate_fmu
from fmpy import read_model_description
import fmpy as fmpy

#from pyfmi import load_fmu
#from pyfmi.fmi import FMUException

from itertools import cycle
from importlib_metadata import version   # included in future Python 3.8

# Set the environment - for Linux a JSON-file in the FMU is read
if platform.system() == 'Linux': locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#------------------------------------------------------------------------------------------------------------------
#  Setup application FMU
#------------------------------------------------------------------------------------------------------------------
      
# Provde the right FMU and load for different platforms in user dialogue:
global fmu_model, model_description
if platform.system() == 'Windows':
   print('Windows - run FMU pre-compiled JModelica 2.14')
   fmu_model ='BPL_TEST2_Perfusion_windows_jm_cs.fmu'        
   model_description = read_model_description(fmu_model)  
   flag_vendor = 'JM'
   flag_type = 'CS' 
elif platform.system() == 'Linux':
#   flag_vendor = input('Linux - run FMU from JModelica (JM) or OpenModelica (OM)?')  
#   flag_type = input('Linux - run FMU-CS (CS) or ME (ME)?')  
#   print()   
   flag_vendor = 'OM'
   flag_type = 'ME'
   if flag_vendor in ['OM','om']:
      print('Linux - run FMU pre-comiled OpenModelica 1.21.0') 
      if flag_type in ['CS','cs']:         
         fmu_model ='BPL_TEST2_Perfusion_linux_om_cs.fmu'    
         model_description = read_model_description(fmu_model)  
      if flag_type in ['ME','me']:         
         fmu_model ='BPL_TEST2_Perfusion_linux_om_me.fmu'    
         model_description = read_model_description(fmu_model)  
   else:    
      print('There is no FMU for this platform')

# Provide various opts-profiles
#if flag_type in ['CS', 'cs']:
#   opts_std = model.simulate_options()
#   opts_std['silent_mode'] = True
#   opts_std['ncp'] = 500 
#   opts_std['result_handling'] = 'binary'     
#elif flag_type in ['ME', 'me']:
#   opts_std = model.simulate_options()
#   opts_std["CVode_options"]["verbosity"] = 50 
#   opts_std['ncp'] = 500 
#   opts_std['result_handling'] = 'binary'
#   opts_std['result_handling'] = 'binary' 
#   opts_std['CVode_options']['atol'] = np.array([1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06])  
#   opts_std['CVode_options']['rtol'] = 0.0001   
#else:    
#   print('There is no FMU for this platform')
  
# Provide various MSL and BPL versions
if flag_vendor in ['JM', 'jm']:
   constants = [v for v in model_description.modelVariables if v.causality == 'local'] 
   MSL_usage = [x[1] for x in [(constants[k].name, constants[k].start) \
                     for k in range(len(constants))] if 'MSL.usage' in x[0]][0]   
   MSL_version = [x[1] for x in [(constants[k].name, constants[k].start) \
                       for k in range(len(constants))] if 'MSL.version' in x[0]][0]
   BPL_version = [x[1] for x in [(constants[k].name, constants[k].start) \
                       for k in range(len(constants))] if 'BPL.version' in x[0]][0] 
elif flag_vendor in ['OM', 'om']:
   MSL_usage = '3.2.3 - used components: RealInput, RealOutput, CombiTimeTable, Types' 
   MSL_version = '3.2.3'
   BPL_version = 'Bioprocess Library version 2.1.1-beta' 
else:    
   print('There is no FMU for this platform')
    
# Simulation time
global simulationTime; simulationTime = 60.0

# Dictionary of time discrete states
timeDiscreteStates = {} 

# Define a minimal compoent list of the model as a starting point for describe('parts')
component_list_minimum = ['bioreactor', 'bioreactor.culture']

#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: stateDict, parDict, diagrams, newplot(), describe()
#------------------------------------------------------------------------------------------------------------------
   
# Create stateDict that later will be used to store final state and used for initialization in 'cont':
#stateDict = model.get_states_list()
global stateDict

# Create dictionaries parDict and parLocation
global parDict; parDict = {}
parDict['V_0'] = 1.0
parDict['VX_0'] = 2.0
parDict['VS_0'] = 100.0

parDict['Y'] = 0.5
parDict['qSmax'] = 1.0
parDict['Ks'] = 0.1

eps=0.05

parDict['filter_eps'] = eps             # Fraction filtrate flow
parDict['filter_alpha_X'] = eps         # Fraction biomass in filtrate flow
parDict['filter_alpha_S'] = eps         # Fraction substrate in filtrate flow

parDict['V_0'] = 100.0
parDict['S_in'] = 30.0

parDict['harvesttank_V_0'] = 0.0
parDict['harvesttank_X_0'] = 0.0
parDict['harvesttank_S_0'] = 0.0

parDict['pump1_t0'] = 0.0
parDict['pump1_F0'] = 0.0
parDict['pump1_t1'] = 17.0
parDict['pump1_F1'] = 0.2/eps
parDict['pump1_t2'] = 50.0
parDict['pump1_F2'] = 0.2/eps
parDict['pump1_t3'] = 993.0
parDict['pump1_F3'] = 0.2/eps
parDict['pump1_t4'] = 994.0
parDict['pump1_F4'] = 0.2/eps

parDict['pump2_t0'] = 0.0
parDict['pump2_F0'] = 0.0
parDict['pump2_t1'] = 17.0
parDict['pump2_F1'] = 0.2/eps
parDict['pump2_t2'] = 50.0
parDict['pump2_F2'] = 0.2/eps
parDict['pump2_t3'] = 993.0
parDict['pump2_F3'] = 0.2/eps
parDict['pump2_t4'] = 994.0
parDict['pump2_F4'] = 0.2/eps

global parLocation; parLocation = {}
parLocation['V_0'] = 'bioreactor.V_0'
parLocation['VX_0'] = 'bioreactor.m_0[1]' 
parLocation['VS_0'] = 'bioreactor.m_0[2]' 

parLocation['Y'] = 'bioreactor.culture.Y'
parLocation['qSmax'] = 'bioreactor.culture.qSmax'
parLocation['Ks'] = 'bioreactor.culture.Ks'

parLocation['filter_eps'] = 'filter.eps'
parLocation['filter_alpha_X'] = 'filter.alpha[1]'
parLocation['filter_alpha_S'] = 'filter.alpha[2]'

parLocation['V_0'] = 'feedtank.V_0'
parLocation['S_in'] = 'feedtank.c_in[2]'

parLocation['harvesttank_V_0'] = 'harvesttank.V_0'
parLocation['harvesttank_X_0'] = 'harvesttank.m_0[1]'
parLocation['harvesttank_S_0'] = 'harvesttank.m_0[2]'

parLocation['pump1_t0'] = 'schemePump1.table[1,1]'
parLocation['pump1_F0'] = 'schemePump1.table[1,2]'
parLocation['pump1_t1'] = 'schemePump1.table[2,1]'
parLocation['pump1_F1'] = 'schemePump1.table[2,2]'
parLocation['pump1_t2'] = 'schemePump1.table[3,1]'
parLocation['pump1_F2'] = 'schemePump1.table[3,2]'
parLocation['pump1_t3'] = 'schemePump1.table[4,1]'
parLocation['pump1_F3'] = 'schemePump1.table[4,2]'
parLocation['pump1_t4'] = 'schemePump1.table[5,1]'
parLocation['pump1_F4'] = 'schemePump1.table[5,2]'

parLocation['pump2_t0'] = 'schemePump2.table[1,1]'
parLocation['pump2_F0'] = 'schemePump2.table[1,2]'
parLocation['pump2_t1'] = 'schemePump2.table[2,1]'
parLocation['pump2_F1'] = 'schemePump2.table[2,2]'
parLocation['pump2_t2'] = 'schemePump2.table[3,1]'
parLocation['pump2_F2'] = 'schemePump2.table[3,2]'
parLocation['pump2_t3'] = 'schemePump2.table[4,1]'
parLocation['pump2_F3'] = 'schemePump2.table[4,2]'
parLocation['pump2_t4'] = 'schemePump2.table[5,1]'
parLocation['pump2_F4'] = 'schemePump2.table[5,2]'

# Extended list of parameters and variables only for describe and not change
global key_variables; key_variables = []
parLocation['mu'] = 'bioreactor.culture.mu'; key_variables.append(parLocation['mu'])

key_variables.append('filter.inlet.c[1]')
key_variables.append('filter.inlet.F')
key_variables.append('filter.filtrate.F')
key_variables.append('filter.retentate.c[1]')
key_variables.append('filter.retentate.F')
key_variables.append('harvesttank.inlet.c[1]')
key_variables.append('harvesttank.inlet.F')

parLocation['feedtank.V'] = 'feedtank.V'; key_variables.append(parLocation['feedtank.V'])
parLocation['feedtank.c_in[2]'] = 'feedtank.c_in[2]'; key_variables.append(parLocation['feedtank.c_in[2]'])

parLocation['V'] = 'bioreactor.V'; key_variables.append(parLocation['V'])
parLocation['VX'] = 'bioreactor.m[1]'; key_variables.append(parLocation['VX'])
parLocation['VS'] = 'bioreactor.m[2]'; key_variables.append(parLocation['VS'])

parLocation['harvesttank.V'] = 'harvesttank.V'; key_variables.append(parLocation['harvesttank.V'])
parLocation['harvesttank.m[1]'] = 'harvesttank.m[1]'; key_variables.append(parLocation['harvesttank.m[1]'])
parLocation['harvesttank.m[2]'] = 'harvesttank.m[2]'; key_variables.append(parLocation['harvesttank.m[2]'])

# Parameter value check 
global parCheck; parCheck = []
parCheck.append("parDict['Y'] > 0")
parCheck.append("parDict['qSmax'] > 0")
parCheck.append("parDict['Ks'] > 0")
parCheck.append("parDict['V_0'] > 0")
parCheck.append("parDict['VX_0'] >= 0")
parCheck.append("parDict['VS_0'] >= 0")
parCheck.append("parDict['pump1_t0'] < parDict['pump1_t1']")
parCheck.append("parDict['pump1_t1'] < parDict['pump1_t2']")
parCheck.append("parDict['pump1_t2'] < parDict['pump1_t3']")
parCheck.append("parDict['pump1_t3'] < parDict['pump1_t4']")
parCheck.append("parDict['pump2_t0'] < parDict['pump2_t1']")
parCheck.append("parDict['pump2_t1'] < parDict['pump2_t2']")
parCheck.append("parDict['pump2_t2'] < parDict['pump2_t3']")
parCheck.append("parDict['pump2_t3'] < parDict['pump2_t4']")

# Create list of diagrams to be plotted by simu()
global diagrams
diagrams = []

# Define standard plots
def newplot(title='Perfusion cultivation', plotType='TimeSeries'):
   """ Standard plot window 
         title = '' """

   # Transfer of argument to global variable
   global ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8
    
   # Reset pens
   setLines()

   # Plot diagram 
   if plotType == 'TimeSeries':

      plt.figure()
      ax1 = plt.subplot(5,1,1)
      ax2 = plt.subplot(5,1,2)
      ax3 = plt.subplot(5,1,3)
      ax4 = plt.subplot(5,1,4)
      ax5 = plt.subplot(5,1,5)
   
      ax1.grid()
      ax1.set_title(title)
      ax1.set_ylabel('S [g/L]')

      ax2.grid()
      ax2.set_ylabel('X [g/L]')

      ax3.grid()
      ax3.set_ylabel('FX [g/h]')

      ax4.grid()
      ax4.set_ylabel('D, mu [1/h]')           

      ax5.grid()
      ax5.set_ylabel('F1, F2 [L/h]')

      ax5.set_xlabel('Time [h]')

      diagrams.clear()
      diagrams.append("ax1.plot(sim_res['time'],sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time'],sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")
      diagrams.append("ax3.plot(sim_res['time'],sim_res['harvesttank.inlet.F']*sim_res['harvesttank.inlet.c[1]'],color='b',linestyle=linetype)")
      diagrams.append("ax3.plot([0, simulationTime], [cstrProdMax(), cstrProdMax()], color='r',linestyle=linetype)")
      diagrams.append("ax3.legend(['FX', 'cstr FX_max'])")        
      diagrams.append("ax4.plot(sim_res['time'],sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)")
      diagrams.append("ax4.plot(sim_res['time'],sim_res['D'],color='b',linestyle=linetype)")  
      diagrams.append("ax4.legend(['mu', 'D'])")    
      diagrams.append("ax5.plot(sim_res['time'],sim_res['feedtank.Fsp'],color='r',linestyle=linetype)")
      diagrams.append("ax5.plot(sim_res['time'],sim_res['filter.Fsp'],color='b',linestyle=linetype)")
      diagrams.append("ax5.legend(['F1', 'F2'])")    


   # Plot diagram 
   elif plotType == 'TimeSeries2':

      plt.figure()
      ax1 = plt.subplot(8,1,1)
      ax2 = plt.subplot(8,1,2)
      ax3 = plt.subplot(8,1,3)
      ax4 = plt.subplot(8,1,4)
      ax5 = plt.subplot(8,1,5)
      ax6 = plt.subplot(8,1,6)
      ax7 = plt.subplot(8,1,7)
      ax8 = plt.subplot(8,1,8)    

      ax1.grid()
      ax1.set_title(title)
      ax1.set_ylabel('S [g/L]')

      ax2.grid()
      ax2.set_ylabel('X [g/L]')

      ax3.grid()
      ax3.set_ylabel('FX [g/h]')

      ax4.grid()
      ax4.set_ylabel('mu [1/h]')           

      ax5.grid()
      ax5.set_ylabel('F1 [L/h]')

      ax6.grid()
      ax6.set_ylabel('F2 [L/h]')

      ax7.grid()
      ax7.set_ylabel('V reactor [L]')

      ax8.grid()
      ax8.set_ylabel('V harvest [L]')

      ax8.set_xlabel('Time [h]')

      diagrams.clear()
      diagrams.append("ax1.plot(sim_res['time'],sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time'],sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")
      diagrams.append("ax3.plot(sim_res['time'],sim_res['harvesttank.inlet.F']*sim_res['harvesttank.inlet.c[1]'],color='b',linestyle=linetype)")
      diagrams.append("ax4.plot(sim_res['time'],sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)")
      diagrams.append("ax5.plot(sim_res['time'],sim_res['bioreactor.inlet[1].F'],color='b',linestyle=linetype)")
      diagrams.append("ax6.plot(sim_res['time'],sim_res['filter.inlet.F'],color='b',linestyle=linetype)")
      diagrams.append("ax7.plot(sim_res['time'],sim_res['bioreactor.V'],color='b',linestyle=linetype)")
      diagrams.append("ax8.plot(sim_res['time'],sim_res['harvesttank.V'],color='b',linestyle=linetype)")

# Define and extend describe for the current application
def describe(name, decimals=3):
   """Look up description of culture, media, as well as parameters and variables in the model code"""
        
   if name == 'culture':
      print('Simplified text book model - only substrate S and cell concentration X')      
 
   elif name in ['broth', 'liquidphase', 'media']: 
      """Describe medium used"""
      X = model_get('liquidphase.X')
      X_description = model_get_variable_description('liquidphase.X') 
      X_mw = model_get('liquidphase.mw[1]')
         
      S = model_get('liquidphase.S') 
      S_description = model_get_variable_description('liquidphase.S')
      S_mw = model_get('liquidphase.mw[2]')
         
      print()
      print('Reactor broth substances included in the model')
      print()
      print(X_description, '    index = ', X, 'molecular weight = ', X_mw, 'Da')
      print(S_description, 'index = ', S, 'molecular weight = ', S_mw, 'Da')
  
   elif name in ['parts']:
      describe_parts(component_list_minimum)
      
   elif name in ['MSL']:
      describe_MSL()

   elif name in ['cstrProdMax']:
      print(cstrProdMax.__doc__,':',cstrProdMax(), '[ g/h ]')

   else:
      describe_general(name, decimals)
 
def cstrProdMax():
    """Calculate from the model maximal chemostat productivity FX_max"""      
    X_max = model_get('bioreactor.culture.Y')*model_get('feedtank.c_in[2]')
    mu_max = model_get('bioreactor.culture.Y')*model_get('bioreactor.culture.qSmax')
    V_nom = model_get('bioreactor.V_0')
    FX_max = mu_max*X_max*V_nom       
    return FX_max

#------------------------------------------------------------------------------------------------------------------
#  General code 
FMU_explore = 'FMU-explore for FMPy version 0.9.7c'
#------------------------------------------------------------------------------------------------------------------

# Define function par() for parameter update
def par(parDict=parDict, parCheck=parCheck, parLocation=parLocation, *x, **x_kwarg):
   """ Set parameter values if available in the predefined dictionaryt parDict. """
   x_kwarg.update(*x)
   x_temp = {}
   for key in x_kwarg.keys():
      if key in parDict.keys():
         x_temp.update({key: x_kwarg[key]})
      else:
         print('Error:', key, '- seems not an accessible parameter - check the spelling')
   parDict.update(x_temp)
   
   parErrors = [requirement for requirement in parCheck if not(eval(requirement))]
   if not parErrors == []:
      print('Error - the following requirements do not hold:')
      for index, item in enumerate(parErrors): print(item)

# Define function init() for initial values update
def init(parDict=parDict, *x, **x_kwarg):
   """ Set initial values and the name should contain string '_0' to be accepted.
       The function can handle general parameter string location names if entered as a dictionary. """
   x_kwarg.update(*x)
   x_init={}
   for key in x_kwarg.keys():
      if '_0' in key: 
         x_init.update({key: x_kwarg[key]})
      else:
         print('Error:', key, '- seems not an initial value, use par() instead - check the spelling')
   parDict.update(x_init)

# Define fuctions similar to pyfmi model.get(), model.get_variable_descirption(), model.get_variable_unit()
def model_get(parLoc, model_description=model_description):
   """ Function corresponds to pyfmi model.get() but returns just a value and not a list"""
   par_var = model_description.modelVariables
   for k in range(len(par_var)):
      if par_var[k].name == parLoc:
         try:
            if par_var[k].name in start_values.keys():
                  value = start_values[par_var[k].name]
            elif par_var[k].variability in ['constant', 'fixed']:        
                  value = float(par_var[k].start)     
            elif par_var[k].variability == 'continuous':
               try:
                  timeSeries = sim_res[par_var[k].name]
                  value = timeSeries[-1]
               except (AttributeError, ValueError):
                  value = None
                  print('Variable not logged')
            else:
               value = None
         except NameError:
            print('Error: Information available after first simution')
            value = None
   return value

def model_get_variable_description(parLoc, model_description=model_description):
   """ Function corresponds to pyfmi model.get_variable_description() but returns just a value and not a list"""
   par_var = model_description.modelVariables
#   value = [x[1] for x in [(par_var[k].name, par_var[k].description) for k in range(len(par_var))] if parLoc in x[0]]
   value = [x.description for x in par_var if parLoc in x.name]   
   return value[0]
   
def model_get_variable_unit(parLoc, model_description=model_description):
   """ Function corresponds to pyfmi model.get_variable_unit() but returns just a value and not a list"""
   par_var = model_description.modelVariables
#   value = [x[1] for x in [(par_var[k].name, par_var[k].unit) for k in range(len(par_var))] if parLoc in x[0]]
   value = [x.unit for x in par_var if parLoc in x.name]
   return value[0]
      
# Define function disp() for display of initial values and parameters
def disp(name='', decimals=3, mode='short'):
   """ Display intial values and parameters in the model that include "name" and is in parLocation list.
       Note, it does not take the value from the dictionary par but from the model. """
   
   def dict_reverser(d):
      seen = set()
      return {v: k for k, v in d.items() if v not in seen or seen.add(v)}
   
   if mode in ['short']:
      k = 0
      for Location in [parLocation[k] for k in parDict.keys()]:
         if name in Location:
            if type(model_get(Location)) != np.bool_:
               print(dict_reverser(parLocation)[Location] , ':', np.round(model_get(Location),decimals))
            else:
               print(dict_reverser(parLocation)[Location] , ':', model_get(Location))               
         else:
            k = k+1
      if k == len(parLocation):
         for parName in parDict.keys():
            if name in parName:
               if type(model_get(Location)) != np.bool_:
                  print(parName,':', np.round(model_get(parLocation[parName]),decimals))
               else: 
                  print(parName,':', model_get(parLocation[parName])[0])

   if mode in ['long','location']:
      k = 0
      for Location in [parLocation[k] for k in parDict.keys()]:
         if name in Location:
            if type(model_get(Location)) != np.bool_:       
               print(Location,':', dict_reverser(parLocation)[Location] , ':', np.round(model_get(Location),decimals))
         else:
            k = k+1
      if k == len(parLocation):
         for parName in parDict.keys():
            if name in parName:
               if type(model_get(Location)) != np.bool_:
                  print(parLocation[parName], ':', dict_reverser(parLocation)[Location], ':', parName,':', 
                     np.round(model_get(parLocation[parName]),decimals))

# Line types
def setLines(lines=['-','--',':','-.']):
   """Set list of linetypes used in plots"""
   global linecycler
   linecycler = cycle(lines)

# Show plots from sim_res, just that
def show(diagrams=diagrams):
   """Show diagrams chosen by newplot()"""
   # Plot pen
   linetype = next(linecycler)    
   # Plot diagrams 
   for command in diagrams: eval(command)

# Define simulation
def simu(simulationTime=simulationTime, mode='Initial', diagrams=diagrams, output_interval=None):
   global sim_res, prevFinalTime, stateDict, stateDictInitial, stateDictInitialLoc, start_values
   
   def extract_variables(diagrams):
       output = []
       variables = [v for v in model_description.modelVariables if v.causality == 'local']
       for j in range(len(diagrams)):
           for k in range(len(variables)):
               if variables[k].name in diagrams[j]:
                   output.append(variables[k].name)
       return output

   # Run simulation
   if mode in ['Initial', 'initial', 'init']: 
      
      start_values = {parLocation[k]:parDict[k] for k in parDict.keys()}
      
      # Simulate
      sim_res = simulate_fmu(
         filename = fmu_model,
         validate = False,
         start_time = 0,
         stop_time = simulationTime,
         output_interval = output_interval,
         record_events = True,
         start_values = start_values,
         fmi_call_logger = None,
         output = list(set(extract_variables(diagrams) + key_variables))
      )
      
   elif mode in ['Continued', 'continued', 'cont']:
      
      # Update parDictMod and create parLocationMod
      try:
         parDictRed = parDict.copy()
         parLocationRed = parLocation.copy()
         for key in parDict.keys():
            if parLocation[key] in stateDictInitial.values(): 
               del parDictRed[key]  
               del parLocationRed[key]
         parLocationMod = dict(list(parLocationRed.items()) + list(stateDictInitialLoc.items()))
      
         # Create parDictMod and parLocationMod
         parDictMod = dict(list(parDictRed.items()) + 
            [(stateDictInitial[key], stateDict[key]) for key in stateDict.keys()])      
      except NameError:
         print("Simulation is first done with default mode='init'")
         prevFinalTime = 0
  
      # Simulate
      sim_res = simulate_fmu(
         filename = fmu_model,
         validate = False,
         start_time = prevFinalTime,
         stop_time = prevFinalTime + simulationTime,
         output_interval = output_interval,
         record_events = True,
         start_values = {parLocationMod[k]:parDictMod[k] for k in parDictMod.keys()},
         fmi_call_logger = None,
         output = list(set(extract_variables(diagrams) + key_variables))
      )

   else:
      print("Error: simulation mode not correct")

   # Plot diagrams from simulation
   linetype = next(linecycler)    
   for command in diagrams: eval(command)
   
   # Create once dictionaries related to handling the initial states
   try: stateDict
   except NameError:
      # Creeate stateDict firt time
      continuous_states = []
      for variable in model_description.modelVariables:
         if variable.derivative is not None: 
            continuous_states.append(variable.derivative.name)
      stateDict = {key:None for key in continuous_states}  
      stateDict.update(timeDiscreteStates)  
      
      # Create stateDictInitial first time
      stateDictInitial = {}
      for key in stateDict.keys():
          if not key[-1] == ']':
               if key[-3:] == 'I.y':
                  stateDictInitial[key] = key[:-10]+'I_0'
               elif key[-3:] == 'D.x':
                  stateDictInitial[key] = key[:-10]+'D_0'
               else:
                  stateDictInitial[key] = key+'_0'
          elif key[-3] == '[':
              stateDictInitial[key] = key[:-3]+'_0'+key[-3:]
          elif key[-4] == '[':
              stateDictInitial[key] = key[:-4]+'_0'+key[-4:]
          elif key[-5] == '[':
              stateDictInitial[key] = key[:-5]+'_0'+key[-5:] 
          else:
              print('The state vector has more than 1000 states')
              break
      
      # Create stateDictInitialLoc first time
      stateDictInitialLoc = {}
      for value in stateDictInitial.values():
          stateDictInitialLoc[value] = value
      
   # Store final state values in stateDict:        
   for key in list(stateDict.keys()):
      stateDict[key] = model_get(key)  
         
   # Store time from where simulation will start next time
   prevFinalTime = sim_res['time'][-1]
         
# Describe model parts of the combined system
def describe_parts(component_list=[]):
   """List all parts of the model""" 
       
   def model_component(variable_name):
      i = 0
      name = ''
      finished = False
      if not variable_name[0] == '_':
         while not finished:
            name = name + variable_name[i]
            if i == len(variable_name)-1:
                finished = True 
            elif variable_name[i+1] in ['.', '(']: 
                finished = True
            else: 
                i=i+1
      if name in ['der', 'temp_1', 'temp_2', 'temp_3', 'temp_4', 'temp_5', 'temp_6', 'temp_7']: name = ''
      return name
    
#   variables = list(model.get_model_variables().keys())
   variables = [v.name for v in model_description.modelVariables]
        
   for i in range(len(variables)):
      component = model_component(variables[i])
      if (component not in component_list) \
      & (component not in ['','BPL', 'Customer', 'today[1]', 'today[2]', 'today[3]', 'temp_2', 'temp_3']):
         component_list.append(component)
      
   print(sorted(component_list, key=str.casefold))

# Describe MSL   
def describe_MSL(flag_vendor=flag_vendor):
   """List MSL version and components used"""
   print('MSL:', MSL_usage)
 
# Describe parameters and variables in the Modelica code
def describe_general(name, decimals):
  
   if name == 'time':
      description = 'Time'
      unit = 'h'
      print(description,'[',unit,']')
      
   elif name in parLocation.keys():
      description = model_get_variable_description(parLocation[name])
      value = model_get(parLocation[name])
      try:
         unit = model_get_variable_unit(parLocation[name])
      except FMUException:
         unit =''
      if unit =='':
         if type(value) != np.bool_:
            print(description, ':', np.round(value, decimals))
         else:
            print(description, ':', value)            
      else:
        print(description, ':', np.round(value, decimals), '[',unit,']')
                  
   else:
      description = model_get_variable_description(name)
      value = model_get(name)
      try:
         unit = model_get_variable_unit(name)
      except FMUException:
         unit =''
      if unit =='':
         if type(value) != np.bool_:
            print(description, ':', np.round(value, decimals))
         else:
            print(description, ':', value)     
      else:
         print(description, ':', np.round(value, decimals), '[',unit,']')
         
# Describe framework
def BPL_info():
   print()
   print('Model for bioreactor has been setup. Key commands:')
   print(' - par()       - change of parameters and initial values')
   print(' - init()      - change initial values only')
   print(' - simu()      - simulate and plot')
   print(' - newplot()   - make a new plot')
   print(' - show()      - show plot from previous simulation')
   print(' - disp()      - display parameters and initial values from the last simulation')
   print(' - describe()  - describe culture, broth, parameters, variables with values/units')
   print()
   print('Note that both disp() and describe() takes values from the last simulation')
   print()
   print('Brief information about a command by help(), eg help(simu)') 
   print('Key system information is listed with the command system_info()')

def system_info():
   """Print system information"""
#   FMU_type = model.__class__.__name__
   constants = [v for v in model_description.modelVariables if v.causality == 'local']
   
   print()
   print('System information')
   print(' -OS:', platform.system())
   print(' -Python:', platform.python_version())
   try:
       scipy_ver = scipy.__version__
       print(' -Scipy:',scipy_ver)
   except NameError:
       print(' -Scipy: not installed in the notebook')
   print(' -FMPy:', version('fmpy'))
   print(' -FMU by:', read_model_description(fmu_model).generationTool)
   print(' -FMI:', read_model_description(fmu_model).fmiVersion)
   if model_description.modelExchange is None:
      print(' -Type: CS')
   else:
      print(' -Type: ME')
   print(' -Name:', read_model_description(fmu_model).modelName)
   print(' -Generated:', read_model_description(fmu_model).generationDateAndTime)
   print(' -MSL:', MSL_version)    
   print(' -Description:', BPL_version)   
   print(' -Interaction:', FMU_explore)
   
#------------------------------------------------------------------------------------------------------------------
#  Startup
#------------------------------------------------------------------------------------------------------------------

BPL_info()