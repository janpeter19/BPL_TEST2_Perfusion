# setup application functions BPL_TEST2_Perfusion, dependent on previous import of functions from fmu_explore 
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2026-08-27 - Created
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#  Specific application functions: newplot(), describe()
#------------------------------------------------------------------------------------------------------------------

# Define standard plots
def newplot(title='Perfusion cultivation', plotType='TimeSeries'):
   """ Standard plot window 
         title = '' """
    
   # Reset pens
   resetPen()

   # Plot diagram 
   if plotType == 'TimeSeries':

      ax1 = plt.subplot(5,1,1)
      ax2 = plt.subplot(5,1,2)
      ax3 = plt.subplot(5,1,3)
      ax4 = plt.subplot(5,1,4)
      ax5 = plt.subplot(5,1,5)
      
      ax.clear()
      ax.append(ax1)
      ax.append(ax2)
      ax.append(ax3)
      ax.append(ax4)
      ax.append(ax5)
   
      ax[0].grid()
      ax[0].set_title(title)
      ax[0].set_ylabel('S [g/L]')

      ax[1].grid()
      ax[1].set_ylabel('X [g/L]')

      ax[2].grid()
      ax[2].set_ylabel('FX [g/h]')

      ax[3].grid()
      ax[3].set_ylabel('D, mu [1/h]')           

      ax[4].grid()
      ax[4].set_ylabel('F1, F2 [L/h]')

      ax[4].set_xlabel('Time [h]')

      diagrams.clear()
      diagrams.append("ax[0].plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax[1].plot(t,sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")
      diagrams.append("ax[2].plot(t,sim_res['harvesttank.inlet.F']*sim_res['harvesttank.inlet.c[1]'],color='b',linestyle=linetype)")
      diagrams.append("ax[2].plot([0, simulationTime], [cstrProdMax(model), cstrProdMax(model)], color='r',linestyle=linetype)")
      diagrams.append("ax[2].legend(['FX', 'cstr FX_max'])")        
      diagrams.append("ax[3].plot(t,sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)")
      diagrams.append("ax[3].plot(t,sim_res['D'],color='b',linestyle=linetype)")  
      diagrams.append("ax[3].legend(['mu', 'D'])")    
      diagrams.append("ax[4].plot(t,sim_res['feedtank.Fsp'],color='r',linestyle=linetype)")
      diagrams.append("ax[4].plot(t,sim_res['filter.Fsp'],color='b',linestyle=linetype)")
      diagrams.append("ax[4].legend(['F1', 'F2'])")    

   # Plot diagram 
   elif plotType == 'TimeSeries2':

      ax1 = plt.subplot(8,1,1)
      ax2 = plt.subplot(8,1,2)
      ax3 = plt.subplot(8,1,3)
      ax4 = plt.subplot(8,1,4)
      ax5 = plt.subplot(8,1,5)
      ax6 = plt.subplot(8,1,6)
      ax7 = plt.subplot(8,1,7)
      ax8 = plt.subplot(8,1,8) 
      
      ax.clear()
      ax.append(ax1)
      ax.append(ax2)
      ax.append(ax3)
      ax.append(ax4)
      ax.append(ax5)      
      ax.append(ax6)
      ax.append(ax7)
      ax.append(ax8)     

      ax[0].grid()
      ax[0].set_title(title)
      ax[0].set_ylabel('S [g/L]')

      ax[1].grid()
      ax[1].set_ylabel('X [g/L]')

      ax[2].grid()
      ax[2].set_ylabel('FX [g/h]')

      ax[3].grid()
      ax[3].set_ylabel('mu [1/h]')           

      ax[4].grid()
      ax[4].set_ylabel('F1 [L/h]')

      ax[5].grid()
      ax[5].set_ylabel('F2 [L/h]')

      ax[6].grid()
      ax[6].set_ylabel('V reactor [L]')

      ax[7].grid()
      ax[7].set_ylabel('V harvest [L]')

      ax[7].set_xlabel('Time [h]')

      diagrams.clear()
      diagrams.append("ax[0].plot(t,sim_res['bioreactor.c[2]'],color='b',linestyle=linetype)")
      diagrams.append("ax[1].plot(t,sim_res['bioreactor.c[1]'],color='b',linestyle=linetype)")
      diagrams.append("ax[2].plot(t,sim_res['harvesttank.inlet.F']*sim_res['harvesttank.inlet.c[1]'],color='b',linestyle=linetype)")
      diagrams.append("ax[3].plot(t,sim_res['bioreactor.culture.q[1]'],color='r',linestyle=linetype)")
      diagrams.append("ax[4].plot(t,sim_res['bioreactor.inlet[1].F'],color='b',linestyle=linetype)")
      diagrams.append("ax[5].plot(t,sim_res['filter.inlet.F'],color='b',linestyle=linetype)")
      diagrams.append("ax[6].plot(t,sim_res['bioreactor.V'],color='b',linestyle=linetype)")
      diagrams.append("ax[7].plot(t,sim_res['harvesttank.V'],color='b',linestyle=linetype)")

# Define and extend describe for the current application
def describe(name, decimals=3):
   """Look up description of culture, media, as well as parameters and variables in the model code"""
        
   if name == 'culture':
      print('Simplified text book model - only substrate S and cell concentration X')      
 
   elif name in ['broth', 'liquidphase', 'media']: 
      """Describe medium used"""
      X = model.get('liquidphase.X')[0] 
      X_description = model.get_variable_description('liquidphase.X') 
      X_mw = model.get('liquidphase.mw[1]')[0]
         
      S = model.get('liquidphase.S')[0] 
      S_description = model.get_variable_description('liquidphase.S')
      S_mw = model.get('liquidphase.mw[2]')[0]
         
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
      print(cstrProdMax.__doc__,':',cstrProdMax(model), '[ g/h ]')

   else:
      describe_general(name, decimals)
 
#------------------------------------------------------------------------------------------------------------------
#  Startup
#------------------------------------------------------------------------------------------------------------------

FMU_explore_info()

