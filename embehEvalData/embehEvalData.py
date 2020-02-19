from numpy import *
from pandas import *
from pyEvalData import *
from fitFunctions import *
from lmfit import *
import numpy as np
from collections import OrderedDict
from IPython.display import display, Markdown, Latex
from shutil import copyfile



def evaluateFunction_test():
    print('Loaded evaluateFunctionss.py')

def plot_fit_and_save(material, t0=0, t0_0=0.2462, tau1_0=0.140, tau2_0=4, A_0=0.5, B_0=-0.3, sigma=0.060, dofit = 1, vary = 1, vary_time = 1, cutoff = 3, lower_cutoff = -1, modl = 'doubleMartin', nan = 'raise', single =False, separate=False, samples = None, data = None):
    
    data.t0 = t0
    
    display(Markdown('# %s'%material))
    sample = samples[material]
    seq = sequence_plotter(material, 'Fi', samples)
    print(seq)
    data.cList = ['M']
    display(sample['data'])
    
    
    figure(None,(15,4.8))
    if dofit:
        pars = Parameters()
        pars.add('t0', value = t0_0, vary = vary_time)#, min=t0_0-0.020, max=t0_0+0.020)
        pars.add('tau1', value = tau1_0, vary = vary)
        pars.add('tau2', value = tau2_0, vary = vary, min = 0, max = 1e9)
        pars.add('A', value = A_0 , vary = vary)
        pars.add('B', value = B_0 ,vary = vary)
        pars['B'].set(expr='-A')
        
        if modl == 'doubleDaniel':
            
            mod = Model(doubleDecayConv2) 
            
            pars.add('alpha', value=1,vary=False)
            pars.add('sigS', value=sigma/(2*np.sqrt(2*np.log(2))),vary=False)
            pars.add('sigH', value=100/(2*np.sqrt(2*np.log(2))),vary=False)
        
        elif modl == 'doubleMartin':
            
            mod = Model(doubleDecayConvMartin)
            
            pars.add('sigPuPr', value=sigma/(2*np.sqrt(2*np.log(2))), vary=False)
        
        res, params, fit = data.fitScanSequence(seq, mod, pars,
                                                sequenceType='fluence',
                                                norm2one=1,
                                                offsetT0=1,
                                                fitReport = 0,
                                                lastResAsPar=1,
                                                select='x2plot<%d'%cutoff,
                                                showSingle = single,
                                                nan_pol=nan,
                                                xGrid = np.arange(lower_cutoff,cutoff,0.001),
                                                plotSeparate = separate
                                               )
        
        samples[material]['data']['tau1'] = res['M']['tau1']
        samples[material]['data']['tau1Err'] = res['M']['tau1Err']
        samples[material]['data']['tau2'] = res['M']['tau2']
        samples[material]['data']['tau2Err'] = res['M']['tau2Err']
        samples[material]['data']['A'] = res['M']['A']
        samples[material]['data']['AErr'] = res['M']['AErr']
        samples[material]['data']['B'] = res['M']['B']
        samples[material]['data']['BErr'] = res['M']['BErr']
        
        if modl == 'doubleDaniel':
            
            samples[material]['data']['sig'] = res['M']['sigS']
            samples[material]['data']['sigErr'] = res['M']['sigSErr']
        
        elif modl == 'doubleMartin':
            
            samples[material]['data']['sig'] = res['M']['sigPuPr']
            samples[material]['data']['sigErr'] = res['M']['sigPuPr']
   
        samples[material]['data']['t0'] = res['M']['t0']
        samples[material]['data']['t0Err'] = res['M']['t0Err']
        
        xlim(-1,cutoff-min(samples[material]['data']['t0']))
        title(material+' With data cut off @ %d ps'%(cutoff-min(samples[material]['data']['t0'])))
        
        if 'FeGd' in material:
            col = 'k'
        elif 'Co' in material:
            col = 'b'
        elif 'Fe' in material:
            col = 'r'
        elif 'Ni' in material:
            col = 'g'
        else:
            col = 'k'
        
        title(material,color = col)
    else:
        data.plotScanSequence(seq,norm2one=1,xGrid = np.arange(lower_cutoff,cutoff,0.001))
        title(material)
        xlim(lower_cutoff,cutoff)

    ylabel('M(t)')
    legend(loc = 'lower right')
    
    ylim(0.3,1.1)
    xlabel('Delay/ps')
    
    xlim(lower_cutoff, cutoff)
    xlabel('Delay/ps')
    tight_layout()
    show()
    display(samples[material]['data'])

################################################################################################################    
    figure(None,(20,5))

    data.cList = ['E']
    data.plotScanSequence(seq,norm2one=1)

    ylabel('E(t)')
    #xscale('symlog',linthreshx = 1)
    legend(loc = 'right')
    xlim(lower_cutoff,cutoff)
    #ylim(0,1.1)
    xlabel('Delay/ps')
    title(material)
    tight_layout()
    data.cList = ['M']
    
    #savefig('results/DOS_Engineering_Session_3/%s_electric.png'%material, dpi=300)
    show()

 

def measurement_printer(threshold, samples, data):
    measurements = {}
    for name,value in samples.items():
        sequence = value['num']
        if len(sequence) != 0:                                                                            
            print('\033[1m'+'==============================================================')                       
            print('%s'%name+'\033[0m')                                                                              
            properties = []
            
            if len(sequence) == 2:
                sequence = range(sequence[0],sequence[-1]+1)
                print('created sequence from %d to %d'%(sequence[0],sequence[-1]+1))
            
            
            for run in sequence:
                try:                                                                                      
                    fluence = data.getScanData(run)[0]['fluence'][0]
                    mhor = data.getScanData(run)[0]['mhor'][0]
                    mver = data.getScanData(run)[0]['mver'][0]
                    magneticfield = mean(data.getScanData(run)[1]['magneticfield'])
                    points = len(data.getScanData(run)[0])
                    if points >= threshold:                                                               
                        print('%04d'%run, '| %05.2f mJ/cm²'%fluence, '| %03d pts'%points, '| mhor: %01.4f '%mhor, '| mver: %01.4f '%mver, '| magneticfield: %01.0fmT '%magneticfield)              
                        counters = [run,round(fluence,3),mhor,mver,round(magneticfield/2)*2]
                        properties = append(properties,counters)
                    elif points == 1:                                                                     
                        print('%04d'%run, '                           | %05.2f mJ/cm²'%fluence, '| Incomplete: %03d pt'%points) 
                    else:                                         
                    
                        print('%04d'%run, '                           | %05.2f mJ/cm²'%fluence, '| Incomplete: %03d pts'%points)                                                                   
                except:                                                                                   
                    print('Didnt work: #%04d'%run)
            properties = reshape(properties,[int(len(properties)/len(counters)),len(counters)])
            measurements[name] = properties[properties[:,1].argsort()]
    return(measurements)

 
def copy_spec_file(year, month, quarter, name):
    filePath = ''                                                                                               
    specFileExt = '.spec'  

    file = year + '_' + month + '_' + name                                                                      
    from shutil import copyfile                                                                                 
    try:                                                                                                        
        copyfile('N:/data/trMOKE/MBI/Sardana/'+year+'_Q'+quarter+'/'+file+specFileExt, filePath+file+specFileExt)
        print('File was copied.')                                                                                
    except OSError:   
        print('Trying to copy:')
        print('N:/data/trMOKE/MBI/Sardana/'+year+'_Q'+quarter+'/'+file+specFileExt)
        print('You are not online or the file name is incorrect. File was not copied')
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


def samples_generator(measurements, blacklist, samples, data, raspi05 = True, prop = 'fluence'):
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
        
        fields = ['Num','B Field','Fi','Pi','R*A','Pt','Pr','Pa','t','r','a','Fa','tau1','tau1Err','tau2','tau2Err','A','AErr','B','BErr','sig','sigErr']
        meas_num = range(len(combined_runs))
        data_init = zeros([len(meas_num),len(fields)])
        database = DataFrame(data_init,meas_num,fields)
        
        for c,combined_run in enumerate(combined_runs):
            
            row = zeros([len(combined_run[0]),22])
            for i,run_num in enumerate(combined_run[0]):
                
                row[i,1]  = round(mean(data.getScanData(run_num)[1]['magneticfield']),0)   #magneticfield
                row[i,2]  = mean(data.getScanData(run_num)[0]['fluence'])                  #Fi
                row[i,3]  = mean(data.getScanData(run_num)[0]['power'])*1000               #Pi
                row[i,4]  = row[i,3]/row[i,2]                                              #R*A
                if raspi05:
                    row[i,5]  = mean(data.getScanData(run_num)[1]['thorlabsPPM'])*1000     #Pt
                    row[i,6]  = mean(data.getScanData(run_num)[1]['thorlabsPM'])*1000      #Pr
                    row[i,7]  = row[i,3] - row[i,5] - row[i,6]                             #Pa = Pi - Pt - Pr
                    row[i,8]  = row[i,5]/row[i,3]                                          #t
                    row[i,9]  = row[i,6]/row[i,3]                                          #r
                    row[i,10]  = row[i,7]/row[i,3]                                         #a
                    row[i,11] = row[i,7]/row[i,4]                                          #Fa
                else:
                    for j in range(5,12):
                        row[i,j] = 0
                    
            row = mean(row,0)
            database.iloc[c]=row
            database.loc[c,'Num'] = array2string(sort(array(combined_run[0]))[::-1],separator=',')
        samples[name]['data'] = database
    return samples
    
    return samples
def sequence_plotter(material, seq_type, samples, prec = 2):
    sample = samples[material]['data']
    return [[[int(num) for num in sample['Num'][i].strip('[]').split(',')],round(sample[seq_type][i],prec)] for i in range(len(sample))]
    
    
