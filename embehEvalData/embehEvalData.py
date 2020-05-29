from numpy import *
from pandas import *
from pyEvalData import *
from fitFunctions import *
from matplotlib.pyplot import*
from lmfit import *
import numpy as np
from collections import OrderedDict
from IPython.display import display, Markdown, Latex, Math
from shutil import copyfile
from copy import deepcopy



def evaluateFunction_test():
    print('Loaded evaluateFunctionss.py')


def plot_fit_and_save_MOKE(material, samples, data, seq_type = 'Fi', modl = 'laser', norm2 = 1, yErr = 'none',
                          t0_0=0, tau1_0=0.1, tau2_0=10, A_0=1, B_0=-1, FWHM=0.054, dofit = 1, vary = 1, vary_t0 = 1, fit_cutoff = False, lastResAsPar = True, changes= None, demagremag = False,
                          cutoff_l = -1, cutoff_r = 3, cutoff_b = 0, cutoff_t = 1.1, save_name = None, xscale_type = 'symlog', **legend_args):
    data.cList = ['M']
    sample = samples[material]
    seq = sequence_generator(material, seq_type, samples)
    if seq_type == 'Fi' or seq_type == 'Fa':
        seq_type2 = 'fluence'
    display(sample['data'])    
    if cutoff_r == None or fit_cutoff == False:
        xgrid = []
        select = ''
        if dofit == True:
            print("Fitting the entire range")
    else:
        xgrid = []
        select='x2plot<%.3f'%fit_cutoff
    
    
    if dofit:
        if demagremag:
            bound = 0
        else:
            bound = None
        pars = Parameters()
        pars.add('t0', value = t0_0, vary = vary_t0)
        pars.add('tau1', value = tau1_0, vary = vary)
        pars.add('tau2', value = tau2_0, min = bound, vary = vary)
        pars.add('A', value = A_0, vary = vary)
        pars.add('B', value = B_0, max = bound, vary = vary)
        #pars['B'].set(expr='-A')
        if changes is not None:
            for change in changes:
                pars[change[0]] = change[1]
        
        if modl == 'slicing':
            
            mod = Model(doubleDecayDoubleConv2) 
            
            pars.add('alpha', value=1,vary=False)
            pars.add('sigS', value=FWHM/(2*np.sqrt(2*np.log(2))),vary=False)
            pars.add('sigH', value=100/(2*np.sqrt(2*np.log(2))),vary=False)
            
            sig_par = 'sigS'
        
        elif modl == 'laser':
            
            mod = Model(doubleDecaySingleConv,nan_policy = 'propagate',)
            
            pars.add('sig', value=FWHM/(2*np.sqrt(2*np.log(2))), vary=False)
            
            sig_par = 'sig'
        print(seq)
        res, params, seqData = data.fitScanSequence(seq, mod, pars,
                                                sequenceType = seq_type2,
                                                norm2one = norm2,
                                                offsetT0 = 1,
                                                fitReport = 0,
                                                lastResAsPar = lastResAsPar,
                                                select = select,
                                                showSingle = False,
                                                xGrid = xgrid,
                                                plotSeparate = False,
                                                yErr = yErr,
                                                )

        samples[material]['data']['tau1'] = res[data.cList[0]]['tau1']
        samples[material]['data']['tau1Err'] = res[data.cList[0]]['tau1Err']
        samples[material]['data']['tau2'] = res[data.cList[0]]['tau2']
        samples[material]['data']['tau2Err'] = res[data.cList[0]]['tau2Err']
        samples[material]['data']['A'] = res[data.cList[0]]['A']
        samples[material]['data']['AErr'] = res[data.cList[0]]['AErr']
        samples[material]['data']['B'] = res[data.cList[0]]['B']
        samples[material]['data']['BErr'] = res[data.cList[0]]['BErr']
        samples[material]['data']['sig'] = res[data.cList[0]][sig_par]
        samples[material]['data']['sigErr'] = res[data.cList[0]][sig_par+'Err']
        samples[material]['data']['t0'] = res[data.cList[0]]['t0']
        samples[material]['data']['t0Err'] = res[data.cList[0]]['t0Err']
    else:
        seqData, params, nams, labs = data.plotScanSequence(seq, norm2one=norm2,sequenceType=seq_type,yErr = None)

    if 'Gd' in material:
        col = 'k'
    elif 'Co' in material:
        col = 'b'
    elif 'Fe' in material:
        col = 'r'
    elif 'Ni' in material:
        col = 'g'
    else:
        col = 'k'

    title(material[:-4], color = col)       
    xlabel('Delay/ps')      
    ylabel(r'M(t) / $M_0$')
    xlim(cutoff_l,cutoff_r)
    ylim(cutoff_b, cutoff_t)
    if xscale_type == 'symlog':
        xscale(xscale_type, linthreshx = 1)
    else:
        xscale(xscale_type)
    
    if len(legend_args)>0:
        legend(**legend_args)
    
    
    #minorticks_on()
    tight_layout()
    if save_name is not None:
        savefig('%s/Magnetic_%s.png'%(save_name, material), dpi=300)
        savefig('results_evaluation/Magnetic_%s.png'%material[:-4], bbox_inches='tight', dpi = 600)
    show()
    ################################################################################################################
    
    if dofit and vary_t0:
        figure(None,(30, 4))

        plot(samples[material]['data'][seq_type], (samples[material]['data']['t0']-t0_0)*1000)

        xlabel(seq_type)
        ylabel('t0 / fs')
        title(material, color = col)
        tight_layout()
        show()
    display(samples[material]['data'])        
################################################################################################################
    figure() # None,(30, 8)

    data.cList = ['E']
    data.plotScanSequence(seq,norm2one=1,yErr = None)

    xlabel('Delay/ps')
    ylabel('E(t)')
    xscale(xscale_type)
    xlim(cutoff_l,cutoff_r)
    ylim(cutoff_b, cutoff_t)
    legend(loc = 'right')
    tight_layout()
    if save_name:
        savefig('%s/Electric_%s.png'%(save_name, material), dpi = 300)
    show()
    return seqData


def elementcolour(element_name):
    '''
    
    Gd: col = 'k'
    Co: col = 'b'
    Fe: col = 'r'
    Ni: col = 'g'
    else: col = 'k' 
    
    '''
    if 'Gd' in element_name:
        col = 'k'
    elif 'Co' in element_name:
        col = 'b'
        if 'Pt' in element_name:
            linestyle = '--'
        else:
            linestyle = '-'
    elif 'Fe' in element_name:
        col = 'r'
        if 'Pt' in element_name:
            linestyle = '--'
        else:
            linestyle = '-'
    elif 'Ni' in element_name:
        col = 'g'
        if 'Pt' in element_name:
            linestyle = '--'
        else:
            linestyle = '-'
    else:
        col = 'k'
    
    return col, linestyle


    
def copy_spec_file(year, month, quarter, name, copy_file = True):
    filePath = ''                                                                                               
    specFileExt = '.spec'  

    file = year + '_' + month + '_' + name                                                                      
    if copy_file:
        from shutil import copyfile                                                                                 
        try:                                                                                                        
            copyfile('N:/data/trMOKE/MBI/Sardana/'+year+'_Q'+quarter+'/'+file+specFileExt, filePath+file+specFileExt)
            print('File was copied.')                                                                                
        except OSError:   
            print('Trying to copy:')
            print('N:/data/trMOKE/MBI/Sardana/'+year+'_Q'+quarter+'/'+file+specFileExt)
            print('You are not online or the file name is incorrect. File was not copied')
    else:
        print('Did not try to copy the file.') 
    data = spec(file,filePath,specFileExt)
    return data 

    
def copy_staticMOKE_files(year, month, day, user='mb',folder = 'new_2020'):
    from shutil import copyfile                                                                                 
    try:                                                                                                        
        copyfile('N:/data/Sputterlab/Transfer_MOKE/'+folder+'/'+year+month+day+'_'+user, filePath+file+specFileExt)
        print('File was copied.')                                                                                
    except OSError:   
        print('Trying to copy:')
        print('N:/data/trMOKE/MBI/Sardana/'+year+'_Q'+quarter+'/'+file+specFileExt)
        print('You are not online or the file name is incorrect. File was not copied')
    data = spec(file,filePath,specFileExt)
    return staticdata 
    

def measurement_printer(threshold, samples, data, verbose = True, legacy = False):
    measurements = {}
    for name,value in samples.items():
        sequence = value['num']
        if len(sequence) != 0:                                                                            
            if verbose:
                print('\033[1m'+'==============================================================')                       
                print('%s'%name+'\033[0m')                                                                              
            properties = []
            
            if len(sequence) == 2:
                sequence = range(sequence[0],sequence[-1]+1)
                if verbose:
                    print('created sequence from %d to %d'%(sequence[0],sequence[-1]+1))
            
            
            for run in sequence:
                try:                                                                          
                    fluence = data.getScanData(run)[0]['fluence'][0]
                    mhor = data.getScanData(run)[0]['mhor'][0]
                    mver = data.getScanData(run)[0]['mver'][0]
                    if legacy:
                        magneticfield = 0
                    else:
                        magneticfield      = mean(data.getScanData(run)[1]['magneticfield'])
                        magneticfield_std  = std(data.getScanData(run)[1]['magneticfield'])
                        cryo_temp          = mean(data.getScanData(run)[1]['cryo_tempposition'])
                        cryo_temp_std      = std(data.getScanData(run)[1]['cryo_tempposition'])
                        cryo_heater        = mean(data.getScanData(run)[1]['cryo_heaterposition'])
                        cryo_heater_std    = std(data.getScanData(run)[1]['cryo_heaterposition'])
                        cryo_pressure      = mean(data.getScanData(run)[1]['cryo_pressure'])
                        cryo_pressure_std  = std(data.getScanData(run)[1]['cryo_pressure'])
                        
                    points = len(data.getScanData(run)[0])
                    if points >= threshold:                                                               
                        if verbose:
                            display(Latex(r'%04d'%run+
                                  ' $\t|$ %05.2f$\,$mJ/cm²'%fluence+
                                  ' $\t|$ %03d$\,$pts'%points+
                                  ' $\t|$ mhor: %01.3f '%mhor+
                                  ' $\t|$ mver: %01.3f '%mver+
                                  ' $\t|$ B: (%01.0f $\pm$ %01.0f)$\,$mT'%(magneticfield,magneticfield_std)+
                                  ' $\t|$ Cryo: (%01.2f $\pm$ %01.2f)$\,$K, (%01.1f $\pm$ %01.1f)$\,$V, (%d $\pm$ %d)$\,$µbar'%(cryo_temp,cryo_temp_std,cryo_heater,cryo_heater_std,1000000*cryo_pressure,1000000*cryo_pressure_std)
                                ))
                        counters = [run,round(fluence,3),mhor,mver,round(magneticfield/2)*2]
                        properties = append(properties,counters)
                    elif points == 1:                                                                     
                        if verbose:
                            print('%04d'%run, '                           | %05.2f mJ/cm²'%fluence, '| Incomplete: %03d pt'%points) 
                    else:                                         

                        if verbose:
                            print('%04d'%run, '                           | %05.2f mJ/cm²'%fluence, '| Incomplete: %03d pts'%points)                                                                   
                except e:
                    if verbose:
                        print('Didnt work: #%04d'%run)
                        print(e)
            properties = reshape(properties,[int(len(properties)/len(counters)),len(counters)])
            measurements[name] = properties[properties[:,1].argsort()]
    return(measurements) 

def delete_elements_from_table(table,elements):
    for element in elements:
            table = table[table[:,0]!=element]
    return table

    
def combine_runs(runs, prop):
    """TEXT.
        
    Args:
        xxxx                        : xxxx
        
    Returns:
        xxx                         : xxx

    """
    props = {'fluence':1, 'mhor':2, 'mver':3, 'magneticfield':4}
    prop_num = props[prop]
    uniques = sort(unique(runs[:,prop_num]))[::-1]
    mask = array([u==runs[:,prop_num] for u in uniques])
    return [[list(runs[:,0][m].astype(int)),fluence] for (m,fluence) in zip(mask,uniques)]


def samples_generator(measurements, blacklist, samples, data, raspi05 = True, prop = 'fluence', legacy = False):
    """TEXT.
        
    Args:
        masurements                 : xxxx
        blacklist                   : xxxx
        samples                     : xxxx
        data                        : xxxx
        raspi05                     : xxxx
        prop                        : Must be one of 'fluence' (default),
                                      'mhor', 'mver', or 'magneticfield'
        
    Returns:
        xxx                         : xxx

    """

    
    for name, value in samples.items():
        runs = delete_elements_from_table(measurements[name],blacklist)
        combined_runs = combine_runs(runs, prop)
        
        fields = ['Num','B Field','mhor','mver','Fi','Pi','R*A','Pt','Pr','Pa','t','r','a','Fa','tau1','tau1Err','tau2','tau2Err','A','AErr','B','BErr','sig','sigErr']
        meas_num = range(len(combined_runs))
        data_init = zeros([len(meas_num),len(fields)])
        database = DataFrame(data_init,meas_num,fields)
        try:
            print(len(asd['DH_Overlap_Test']['data']['Num'][0]))
        except:
            pass
        for c,combined_run in enumerate(combined_runs):
            
            row = zeros([len(combined_run[0]),24])
            for i,run_num in enumerate(combined_run[0]):
                
                if legacy:
                    row[i,1]  = mean(data.getScanData(run_num)[0]['mhor'])                     #zeros
                else:
                    row[i,1]  = round(mean(data.getScanData(run_num)[1]['magneticfield']),0)   #magneticfield
                row[i,2]  = round(mean(data.getScanData(run_num)[0]['mhor']),3)                #mhor
                row[i,3]  = round(mean(data.getScanData(run_num)[0]['mver']),3)                #mver
                
                row[i,4]  = mean(data.getScanData(run_num)[0]['fluence'])                      #Fi
                row[i,5]  = mean(data.getScanData(run_num)[0]['power'])*1000                   #Pi
                row[i,6]  = row[i,5]/row[i,4]                                                  #Pi/Fi = R*A
                if raspi05:
                    row[i,7]  = mean(data.getScanData(run_num)[1]['thorlabsPPM'])*1000         #Pt
                    row[i,8]  = mean(data.getScanData(run_num)[1]['thorlabsPM'])*1000          #Pr
                    row[i,9]  = row[i,5] - row[i,7] - row[i,8]                                 #Pa = Pi - Pt - Pr
                    row[i,10]  = row[i,7]/row[i,5]                                             #t
                    row[i,11]  = row[i,8]/row[i,5]                                             #r
                    row[i,12]  = row[i,9]/row[i,5]                                             #a
                    row[i,13] = row[i,9]/row[i,6]                                              #Fa
                else:
                    for j in range(7,14):
                        row[i,j] = 0
                    
            row = mean(row,0)
            database.iloc[c]=row
            database.loc[c,'Num'] = array2string(sort(array(combined_run[0]))[::-1],separator=',')
        samples[name]['data'] = database
    return deepcopy(samples)


def sequence_generator(material, seq_type, samples, prec = 2):
    sample = samples[material]['data']
    return [[[int(num) for num in sample['Num'][i].strip('[]').split(',')],round(sample[seq_type][i],prec)] for i in range(len(sample))]
    
