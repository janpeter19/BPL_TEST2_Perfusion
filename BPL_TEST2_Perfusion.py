# setup applicateion data BPL_TEST2_Perfusion
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2026-08-27 - Created
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#  Framework
#------------------------------------------------------------------------------------------------------------------

# Setup framework
import sys
import platform
import locale
import numpy as np
import matplotlib.pyplot as plt 
from pyfmi import load_fmu

# Set the environment - for Linux a JSON-file in the FMU is read
if platform.system() == 'Linux': locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#------------------------------------------------------------------------------------------------------------------
#  Setup application FMU
#------------------------------------------------------------------------------------------------------------------
      
# Provde the right FMU and load for different platforms in user dialogue:
if platform.system() == 'Windows':
   print('Windows - run FMU pre-compiled JModelica 2.14')
   flag_vendor = 'JM'
   flag_type = 'CS'
   fmu_model ='BPL_TEST2_Perfusion_windows_jm_cs.fmu'        
   model = load_fmu(fmu_model, log_level=0)  
elif platform.system() == 'Linux':  
   flag_vendor = 'OM'
   flag_type = 'ME'
   if flag_vendor in ['OM','om']:
      print('Linux - run FMU pre-compiled OpenModelica') 
      if flag_type in ['CS','cs']:         
         fmu_model ='BPL_TEST2_Perfusion_linux_om_cs.fmu'    
         model = load_fmu(fmu_model, log_level=0) 
      if flag_type in ['ME','me']:         
         fmu_model ='BPL_TEST2_Perfusion_linux_om_me.fmu' 
#         fmu_model ='BPL_TEST2_Perfusion_linux_2404_om_me.fmu'       
         model = load_fmu(fmu_model, log_level=0)
   else:    
      print('There is no FMU for this platform')

# Provide various opts-profiles
if flag_type in ['CS', 'cs']:
   opts_std = model.simulate_options()
   opts_std['silent_mode'] = True
   opts_std['ncp'] = 500 
   opts_std['result_handling'] = 'binary'     
elif flag_type in ['ME', 'me']:
   opts_std = model.simulate_options()
   opts_std["CVode_options"]["verbosity"] = 50 
   opts_std['ncp'] = 500 
   opts_std['result_handling'] = 'binary'
   opts_std['result_handling'] = 'binary' 
   opts_std['CVode_options']['atol'] = np.array([1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06])  
   opts_std['CVode_options']['rtol'] = 0.0001   
else:    
   print('There is no FMU for this platform')
  
# Provide various MSL and BPL versions
if flag_vendor in ['JM', 'jm']:
   MSL_usage = model.get('MSL.usage')[0]
   MSL_version = model.get('MSL.version')[0]
   BPL_version = model.get('BPL.version')[0]
elif flag_vendor in ['OM', 'om']:
   MSL_usage = '4.1.0 - used components: RealInput, RealOutput, CombiTimeTable, Types' 
   MSL_version = '4.1.0'
   BPL_version = 'Bioprocess Library version 2.3.2' 
else:    
   print('There is no FMU for this platform')
   
#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: stateValue, parValue, parLocation, parCheck, diagrams, ax, lines
#------------------------------------------------------------------------------------------------------------------   
    
# Simulation time
simulationTime = 60.0
prevFinalTime = 0

# Dictionary of time discrete states
timeDiscreteStates = {} 

# Create stateValue that later will be used to store final state and used for initialization in 'cont':
stateValue =  {}
stateValue = model.get_states_list()
stateValue.update(timeDiscreteStates)

# Define a minimal compoent list of the model as a starting point for describe('parts')
component_list_minimum = ['bioreactor', 'bioreactor.culture']

# Provide process diagram on disk
fmu_process_diagram ='BPL_TEST2_Perfusion_process_diagram_om.png'

# Create dictionaries parValue and parLocation
parValue = {}
parValue['V_start'] = 1.0
parValue['VX_start'] = 2.0
parValue['VS_start'] = 100.0

parValue['Y'] = 0.5
parValue['qSmax'] = 1.0
parValue['Ks'] = 0.1

eps=0.05

parValue['filter_eps'] = eps             # Fraction filtrate flow
parValue['filter_alpha_X'] = eps         # Fraction biomass in filtrate flow
parValue['filter_alpha_S'] = eps         # Fraction substrate in filtrate flow

parValue['V_start'] = 100.0
parValue['S_in'] = 30.0

parValue['harvesttank_V_start'] = 0.0
parValue['harvesttank_X_start'] = 0.0
parValue['harvesttank_S_start'] = 0.0

parValue['pump1_t0'] = 0.0
parValue['pump1_F0'] = 0.0
parValue['pump1_t1'] = 17.0
parValue['pump1_F1'] = 0.2/eps
parValue['pump1_t2'] = 50.0
parValue['pump1_F2'] = 0.2/eps
parValue['pump1_t3'] = 993.0
parValue['pump1_F3'] = 0.2/eps
parValue['pump1_t4'] = 994.0
parValue['pump1_F4'] = 0.2/eps

parValue['pump2_t0'] = 0.0
parValue['pump2_F0'] = 0.0
parValue['pump2_t1'] = 17.0
parValue['pump2_F1'] = 0.2/eps
parValue['pump2_t2'] = 50.0
parValue['pump2_F2'] = 0.2/eps
parValue['pump2_t3'] = 993.0
parValue['pump2_F3'] = 0.2/eps
parValue['pump2_t4'] = 994.0
parValue['pump2_F4'] = 0.2/eps

parLocation = {}
parLocation['V_start'] = 'bioreactor.V_start'
parLocation['VX_start'] = 'bioreactor.m_start[1]' 
parLocation['VS_start'] = 'bioreactor.m_start[2]' 

parLocation['Y'] = 'bioreactor.culture.Y'
parLocation['qSmax'] = 'bioreactor.culture.qSmax'
parLocation['Ks'] = 'bioreactor.culture.Ks'

parLocation['filter_eps'] = 'filter.eps'
parLocation['filter_alpha_X'] = 'filter.alpha[1]'
parLocation['filter_alpha_S'] = 'filter.alpha[2]'

parLocation['V_start'] = 'feedtank.V_start'
parLocation['S_in'] = 'feedtank.c_in[2]'

parLocation['harvesttank_V_start'] = 'harvesttank.V_start'
parLocation['harvesttank_X_start'] = 'harvesttank.m_start[1]'
parLocation['harvesttank_S_start'] = 'harvesttank.m_start[2]'

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

# Extra for describe()
parLocation['mu'] = 'bioreactor.culture.mu'

# Parameter value check - especially for hysteresis to avoid runtime error
parCheck = []
parCheck.append("parValue['Y'] > 0")
parCheck.append("parValue['qSmax'] > 0")
parCheck.append("parValue['Ks'] > 0")
parCheck.append("parValue['V_start'] > 0")
parCheck.append("parValue['VX_start'] >= 0")
parCheck.append("parValue['VS_start'] >= 0")
parCheck.append("parValue['pump1_t0'] < parValue['pump1_t1']")
parCheck.append("parValue['pump1_t1'] < parValue['pump1_t2']")
parCheck.append("parValue['pump1_t2'] < parValue['pump1_t3']")
parCheck.append("parValue['pump1_t3'] < parValue['pump1_t4']")
parCheck.append("parValue['pump2_t0'] < parValue['pump2_t1']")
parCheck.append("parValue['pump2_t1'] < parValue['pump2_t2']")
parCheck.append("parValue['pump2_t2'] < parValue['pump2_t3']")
parCheck.append("parValue['pump2_t3'] < parValue['pump2_t4']")

# Create list of diagrams to be plotted by simu()
diagrams = []

# Create an empty list axes to be defined in newplot() and plotted by simu() or show()
ax = []

# Create list of pens for the diagrams
lines = ['-','--',':','-.']

#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: external function
#------------------------------------------------------------------------------------------------------------------

def cstrProdMax(model):
    """Calculate from the model maximal chemostat productivity FX_max"""      
    X_max = model.get('bioreactor.culture.Y')*model.get('feedtank.c_in[2]')
    mu_max = model.get('bioreactor.culture.Y')*model.get('bioreactor.culture.qSmax')
    V_nom = model.get('bioreactor.V_start')
    FX_max = mu_max*X_max*V_nom       
    return FX_max[0]