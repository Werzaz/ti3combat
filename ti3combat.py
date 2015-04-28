#!/usr/bin/python

"""TI3 Combat Calculator (Development Version 3) by Sunchaser""" 

import numpy as np
import sys
import cPickle as Pickle
from copy import deepcopy

_Base_Values = {'WS': {'Type':'WS', 'Roll':3, 'Dice':3, 'HP':2,
                       'MaxHP':2, 'Cost':12., 'Evasion':0},
                'DN': {'Type':'DN', 'Roll':5, 'Dice':1, 'HP':2,
                       'MaxHP':2, 'Cost':5., 'Evasion':0},
                'CA': {'Type':'CA', 'Roll':7, 'Dice':1, 'HP':1,
                       'MaxHP':1, 'Cost':2., 'Evasion':0},
                'DD': {'Type':'DD', 'Roll':9, 'Dice':1, 'HP':1,
                       'MaxHP':1, 'Cost':1., 'Evasion':0},
                'CV': {'Type':'CV', 'Roll':9, 'Dice':1, 'HP':1,
                       'MaxHP':1, 'Cost':3., 'Evasion':0},
                'FT': {'Type':'FT', 'Roll':9, 'Dice':1, 'HP':1,
                       'MaxHP':1, 'Cost':.5, 'Evasion':0}}

_unit_cost = {'WS': 12., 'DN': 5., 'CA': 2., 'DD': 1., 'CV': 3., 'FT': 0.5}
_unit_available = {'WS': 2, 'DN': 5, 'CA': 8, 'DD': 8, 'CV': 4, 'FT': 1e6}

def binom(n,k): 
    """Binomial coefficient (n k)"""
    return np.prod([float(n-k+i)/i for i in range(1,k+1)])

def dist(n,p): 
    """Binomial distribution 
    for k indepent dice rolls 
    at success propability p"""
    return np.array([[binom(n,k)*(1-p)**(n-k)*p**k 
                      for k in range(n+1)]])

# dists should list of 2d-np-arrays
def convolve(dists):
    """ Discreet convolution of probability distributions 
    with two parameters, e.g. number of Shock Troops and Hits.

    Arguments:
    dists -- 2d numpy array. The sum of all entries should add up to 1.
             Typically first dimension is number of Shock Troops created
             and second dimension is number hits."""  
    out=dists[0]
    for dist in dists[1:]:
        temp=np.zeros((out.shape[0]+dist.shape[0]-1,
                       out.shape[1]+dist.shape[1]-1))
        for k in range(temp.shape[0]):
            ind=[(k1,k2) for k1 in range(out.shape[0]) 
                 for k2 in range(dist.shape[0])
                 if k1+k2==k]
            for l in range(temp.shape[1]):
                ind2=[(l1,l2) for l1 in range(out.shape[1]) 
                     for l2 in range(dist.shape[1])
                     if l1+l2==l]
                for k1,k2 in ind:
                    for l1,l2 in ind2:
                        temp[k,l]+=out[k1,l1]*dist[k2,l2]
        out=temp[:]
    return out

def Probabilities(Rolls,GFs=[],Rerolls=0):
    """Probability distribution for combat including rerolls
    Note that this function will become very slow for large combats.
    It should only be used if rerolls are necessary.
    Otherwise one should use dist() and convolve().

    Arguments:
    Rolls -- List of combat values for each die, ordered in priority of 
             rerolling, i.e. typically lowest first.
             For the Jol-Nar Flagship with combat value k use 'JNFS:k'.

    Keyword arguments:
    GFs     -- Indices of the combat values in Rolls.
    Rerolls -- Number of Rerolls.
    """
    outcomes=[[]]
    JNFSList=['JNFS:%i'%k for k in range(1,11)]
    JNFS=len([a for a in Rolls if a in JNFSList])
    out=np.zeros((1+len(GFs),1+len(Rolls)+2*JNFS))
    for k,Roll in enumerate(Rolls):
        if Roll in JNFSList: add=[0,1,3]
        elif k in GFs: add=[0,1,'S']
        else: add=[0,1]
        outcomes=[a+[b] for a in outcomes for b in add]
    for outcome in outcomes:
        STs=len([a for a in outcome if a=='S'])
        Hits=sum([a for a in outcome if a!='S'])+STs
        p=1
        for Roll,Result in zip(Rolls,outcome):
            if Roll not in JNFSList:
                if Result==1 and k in GFs: p*=1.-Roll/10.
                elif Result==1: p*=1.1-Roll/10.
                elif Result==0: p*=(Roll-1)/10.
                else: p*=.1
            else:
                Roll_J=int(Roll.split(':')[1])
                if Result==1: p*=1.-Roll_J/10.
                elif Result==0: p*=(Roll_J-1)/10.
                else: p*=.1
        Missed=[l for l,b in enumerate(outcome) if b==0]
        if Missed==[] or Rerolls==0:
            out[STs,Hits]+=p
        else:
            outcomes_r=[[]]
            Rolls_r=[Rolls[k] for k in Missed]
            for a,k in zip(range(Rerolls),Missed):
                if Rolls[k] in JNFSList: add=[0,1,3]
                elif k in GFs: add=[0,1,'S']
                else: add=[0,1]
                outcomes_r=[a+[b] for a in outcomes_r for b in add]
            for outcome_r in outcomes_r:
                STs_r=len([a for a in outcome_r if a=='S'])
                Hits_r=sum([a for a in outcome_r if a!='S'])+STs_r  
                p2=1.
                for Roll,Result in zip(Rolls_r,outcome_r):
                    if Roll not in JNFSList:
                        if Result==1 and k in GFs: p2*=1.-Roll/10.
                        elif Result==1: p2*=1.1-Roll/10.
                        elif Result==0: p2*=(Roll-1)/10.
                        else: p2*=.1
                    else:
                        Roll_J=int(Roll.split(':')[1])
                        if Result==1: p2*=1.-Roll_J/10.
                        elif Result==0: p2*=(Roll_J-1)/10.
                        else: p2*=.1                     
                out[STs+STs_r,Hits+Hits_r]+=p*p2
    return out

def BattleKey(Forces,reverse=False):
    """Make a single string containing 
    all information of the Forces dictionary."""
    if Forces['Space']: out='Space:'
    else: out='Ground:'
    if not reverse:
        PlayerKeys = ['Attacker','Defender']
    else:
        PlayerKeys = ['Defender','Attacker']
    for Key in PlayerKeys:
        for SubKey in ['Tech','DmgOrder','RepairOrder']:
            for k,item in enumerate(Forces[Key][SubKey]):
                if k<len(Forces[Key][SubKey])-1: out+=item+';'
                else: out+=item
            out+=','
        for k,Unit in enumerate(sorted(Forces[Key]['Units'].keys())):
            out+=Unit+'|'+Forces[Key]['Units'][Unit]['Type']+'|'
            for SubKey in ['Roll','Dice','HP','MaxHP','Cost']:
                out+=str(Forces[Key]['Units'][Unit][SubKey])+'/'
            out+=str(Forces[Key]['Units'][Unit]['Evasion'])
            if k<len(Forces[Key]['Units'].keys())-1: out+=';'
        out+=','
        for k,Bonus in enumerate(Forces[Key]['Bonus']):
            out+='%i/%i'%(tuple(Bonus))
            if k<len(Forces[Key]['Bonus'])-1: out+=';'
        out+=','
        for k,Special in enumerate(Forces[Key]['Special']):
            out+='/'.join(Special)
            if k<len(Forces[Key]['Special'])-1: out+=';'
        out+=','
        out+=str(Forces[Key]['Rerolls'])
        if Key==PlayerKeys[0]: out+=':'
    return out

def ShortKey(Forces,from_key=False):
    """Make a shorter string containing
    only the units. No inverse function available."""
    if from_key:
        Forces = InverseBattleKey(Forces)
    out='|'.join([','.join([Name+''.join(['*' for k in range(Unit['MaxHP']-Unit['HP'])])
                            for Name, Unit in sorted(Forces[Key]['Units'].items(),
                                                     key=lambda x: x[0])])
                  for Key in ['Attacker','Defender']])
    return out

def InverseBattleKey(BattleKey):
    """Invert BattleKey()."""
    SubKeys=['Roll','Dice','HP','MaxHP','Cost','Evasion']
    out={'Attacker':{},'Defender':{}}
    if BattleKey.split(':')[0]=='Space': out['Space']=True
    else: out['Space']=False
    for Key,item in zip(['Attacker','Defender'],BattleKey.split(':')[1:]):
        for k,SubKey in enumerate(['Tech','DmgOrder','RepairOrder']):
            if item.split(',')[k]=='':
                out[Key][SubKey]=[]
            else:
                out[Key][SubKey]=item.split(',')[k].split(';')
        out[Key]['Units']={}
        if item.split(',')[3]!='':
            for Unit in item.split(',')[3].split(';'):
                temp={'Type':Unit.split('|')[1]}
                for SubKey,Value in zip(SubKeys,Unit.split('|')[2].split('/')):
                    if SubKey=='Cost': temp[SubKey]=float(Value)
                    else: temp[SubKey]=int(Value)
                out[Key]['Units'][Unit.split('|')[0]]=temp
        out[Key]['Bonus']=[]   
        if item.split(',')[4]!='':
            for Bonus in item.split(',')[4].split(';'):
                out[Key]['Bonus']+=[[int(a) for a in Bonus.split('/')]]
        out[Key]['Special']=[]  
        if item.split(',')[5]!='':
            for Special in item.split(',')[5].split(';'):
                out[Key]['Special']+=[Special.split('/')]
        out[Key]['Rerolls']=int(item.split(',')[6])
    return out
    
def Resolve_Round(Forces, Casualties, STs=[0,0],MinGF4ST=0,
                  STCval=[5,5], AdvanceBoni=True):
    """Removes and Casualties from Forces.
    If Mercenaries are among the units, 
    the output will include the Evasion probabilities.
    If there are boni/mali for a specific round,
    that round number is decreased by 1 (or dropped if 0). 

    Arguments:
    Forces -- dictionaries specifying the units in the battle, 
              as well as damage order, repair order (for Duranium Armor) 
              and Technologies.
    Casualties -- list of 2 integers specifying the number of casualties 
                  each side  has to take.

    Keyword arguments:
    STs -- list of 2 integers specfying how many Shock Troops each side gains.
    MinGF4ST -- minimum number of GFs required to accompany Shock Troops
                (default 0 for the House Rules)
    STCval -- Lists of Shock Troop combat values
    AdvanceBoni -- Set to False if Boni should not be changed.
    """
    out=[[deepcopy(Forces),1.,[[],[]]]]
    iter_list=zip([0,1],['Attacker','Defender'],Casualties,STs)
    for m,Key,Hits,STgain in iter_list:
        if 'Runa' in Forces[Key]['Units'].keys(): Hits=max(1,Hits-1)
        n=0
        while n < Hits:
            temp=[[],[],[]]
            for l,outcome in enumerate(out):
                k=0
                if outcome[0][Key]['Units']=={}: 
                    Success = True
                    temp[0]+=[outcome[0]]
                    temp[1]+=[outcome[1]]
                    temp[2]+=[outcome[2]]
                else: Success = False
                while not Success:
                    Target=outcome[0][Key]['DmgOrder'][k]
                    # First check for Merc
                    if outcome[0][Key]['Units'][Target]['Type']=='Merc':
                        # Check if Merc has been assigned a hit yet
                        if Target in outcome[2][m]:
                            Plastic=[a for a 
                                     in outcome[0][Key]['Units'].keys()
                                     if outcome[0][Key]['Units'][
                                    a]['Type']!='Merc']
                            Mercs=[a for a 
                                   in outcome[0][Key]['Units'].keys()
                                   if outcome[0][Key]['Units'][
                                    a]['Type']=='Merc']
                            Already=[outcome[2][m].count(a) for a in Mercs]
                            if (Plastic==[] and 
                                min(Already)-outcome[2][m].count(
                                    Target)>-1):
                                # Damage Merc                            
                                if Target=='52N6':
                                    Split=False
                                    Succes=True
                                    outcome[0][Key]['Units'][
                                        Target]['HP']-=1
                                    if (outcome[0][Key]['Units'][
                                        Target]['HP']==1 and 
                                        'Noneuclidean' in 
                                        outcome[0][Key]['Tech']): n+=1
                                else:
                                    Split=True
                                    Success=True
                                    Roll=outcome[0][Key]['Units'][
                                        Target]['Evasion']
                                    p=1.1-Roll/10.
                            else:
                                k+=1
                        else:
                            # Damage Merc
                            if Target=='52N6':
                                Split=False
                                Succes=True
                                outcome[0][Key]['Units'][
                                    Target]['HP']-=1
                                if (outcome[0][Key]['Units'][
                                        Target]['HP']==1 and 
                                    'Noneuclidean' in 
                                    outcome[0][Key]['Tech']): n+=1
                            else:
                                Split=True
                                Success=True
                                Roll=outcome[0][Key]['Units'][
                                    Target]['Evasion']
                                p=1.1-Roll/10.
                    elif outcome[0][Key]['Units'][Target]['MaxHP']>1:
                        # Check if damaged
                        Dmg=outcome[0][Key]['Units'][Target]['MaxHP']
                        Dmg-=outcome[0][Key]['Units'][Target]['HP']
                        Occ=[]
                        for q,item in enumerate(
                            outcome[0][Key]['DmgOrder']):
                            if item==Target: Occ+=[q]
                        if k!=Occ[Dmg]: k+=1
                        else:
                            # Damage Unit
                            Split=False
                            Success=True
                            outcome[0][Key]['Units'][Target]['HP']-=1
                            if (k==Occ[0] and 'Noneuclidean' in 
                                outcome[0][Key]['Tech']): n+=1
                    else:
                        # Damage Unit
                        Split=False
                        Success=True
                        outcome[0][Key]['Units'][Target]['HP']-=1
                # Remove Unit if dead
                if outcome[0][Key]['Units']!={}:
                    if not Split:
                        if outcome[0][Key]['Units'][Target]['HP']==0:
                            if Target == 'FS_Nekro':
                                outcome[0]['Attacker']['Units']={}
                                outcome[0]['Attacker']['DmgOrder']=[]
                                outcome[0]['Attacker']['RepairOrder']=[]
                                outcome[0]['Defender']['Units']={}
                                outcome[0]['Defender']['DmgOrder']=[]
                                outcome[0]['Defender']['RepairOrder']=[]
                            else:
                                del outcome[0][Key]['Units'][Target]
                                while Target in outcome[0][Key]['DmgOrder']:
                                    outcome[0][Key]['DmgOrder'].remove(Target)
                                while (Target in 
                                       outcome[0][Key]['RepairOrder']):
                                    outcome[0][Key][
                                        'RepairOrder'].remove(Target)
                        temp[0]+=[deepcopy(outcome[0])]
                        temp[1]+=[outcome[1]]
                        temp[2]+=[outcome[2]]
                    else:
                        New=deepcopy(outcome[0])
                        del New[Key]['Units'][Target]
                        while Target in New[Key]['DmgOrder']:
                            New[Key]['DmgOrder'].remove(Target)
                        while Target in New[Key]['RepairOrder']:
                            New[Key]['RepairOrder'].remove(Target)
                        temp[0]+=[New,deepcopy(outcome[0])]
                        temp[1]+=[outcome[1]*(1-p),outcome[1]*p]
                        new_2=deepcopy(outcome[2])
                        new_2[m]+=[Target]
                        temp[2]+=[new_2,new_2]
            out=[]
            for spam, eggs, ham in zip(temp[0], temp[1], temp[2]):
                out += [[spam, eggs, ham]]
            n+=1
        if 'DA' in Forces[Key]['Tech']:
            for outcome in out:
                Rep=deepcopy(outcome[0][Key]['RepairOrder'])
                Repaired=False
                while Rep!=[] and not Repaired:
                    Target=Rep.pop(0)
                    if (outcome[0][Key]['Units'][Target]['HP']<
                        outcome[0][Key]['Units'][Target]['MaxHP']):
                        outcome[0][Key]['Units'][Target]['HP']+=1
                        Repaired=True
        if STgain > 0:
            for outcome in out:
                nGF = len([a for a in outcome[0][Key]['Units'].keys() 
                           if a[:2] == 'GF'])
                nST = len([a for a in outcome[0][Key]['Units'].keys() 
                           if a[:2] == 'ST'])
                if nGF > MinGF4ST:
                    ST_Tmpl = {'Type':'ST', 'Roll':STCval[m], 'Dice':1,
                               'HP':1, 'MaxHP':1, 'Cost':.5, 'Evasion':0}
                    Types = [a[:2] for a in outcome[0][Key]['DmgOrder']]
                    print Types
                    if 'ST' in Types:
                        ST_position = Types.index('ST')
                    elif 'GF' in Types:
                        ST_position = Types.index('GF')
                    #else:
                    #    ST_position = 0
                    for k in range(min(STgain, nGF - MinGF4ST)):
                        del outcome[0][Key]['Units']['GF%i'%(nGF - k)]
                        outcome[0][Key]['DmgOrder'].remove('GF%i'%(nGF - k))
                        outcome[0][Key]['Units']['ST%i'%(nST + k + 1)] = ST_Tmpl
                        newOrder = (outcome[0][Key]['DmgOrder'][:ST_position] +
                                    ['ST%i'%(nST + k + 1)] +
                                    outcome[0][Key]['DmgOrder'][ST_position:])
                        outcome[0][Key]['DmgOrder'] = newOrder
        if AdvanceBoni:
            for outcome in out:
                New_Bonus=[]
                for Bonus in outcome[0][Key]['Bonus']:
                    if Bonus[1] > 1: New_Bonus += [[Bonus[0],Bonus[1]-1]]
                outcome[0][Key]['Bonus'] = New_Bonus
    out0 = []
    for outcome in out:
        # Check for finished battles. Delete remaining Boni if so
        if (outcome[0]['Attacker']['Units']=={} or 
            outcome[0]['Defender']['Units']=={}):
            outcome[0]['Attacker']['Bonus'] = []
            outcome[0]['Defender']['Bonus'] = []
        if outcome[0] not in [a[0] for a in out0]:
            out0 += [outcome[:2]]
        else:
            k = [a[0] for a in out0].index(outcome[0])
            out0[k][1] += outcome[1]
    return out0

def Battle(Forces, FctCalls=0, BattleDict=None):
    """Main function to resolve a battle.

    Arguments:
    Forces -- dictionaries specifying the units in the battle, 
              as well as damage order, repair order (for Duranium Armor) 
              and Technologies.

    Keyword arguments:
    FctCalls   -- number of previous function calls to assess the effort.
    BattleDict -- dictionary for all smaller battles.

    TODO:
    - Ground Battles
    - PDS and anything precombat
    """
    if BattleDict is None:
        BattleDict = {}

    # If Battle previously calculated, return those results:
    if BattleKey(Forces) in BattleDict.keys() :
        return (BattleDict[BattleKey(Forces)], FctCalls+1, BattleDict, 
                ResCost(Forces,BattleDict[BattleKey(Forces)]))

    # Check for Bonus
    if 1 in [a[1] for a in  Forces['Attacker']['Bonus']]:
        k = [a[1] for a in  Forces['Attacker']['Bonus']].index(1)
        ABonus = Forces['Attacker']['Bonus'][k][0]
    else: 
        ABonus = 0
    if 1 in [a[1] for a in  Forces['Defender']['Bonus']]:
        k = [a[1] for a in  Forces['Defender']['Bonus']].index(1)
        DBonus = Forces['Defender']['Bonus'][k][0]
    else:
        DBonus = 0

    # Probability distributions for attacker and defender
    ADist = np.array([[1.]])
    for Key in Forces['Attacker']['Units'].keys():
        next_roll = dist(Forces['Attacker']['Units'][Key]['Dice'],
                         1.1-Forces['Attacker']['Units'][Key]['Roll']/10.
                         +ABonus/10.)
        ADist = convolve([ADist,next_roll])
    DDist = np.array([[1.]])
    for Key in Forces['Defender']['Units'].keys():
        next_roll = dist(Forces['Defender']['Units'][Key]['Dice'],
                         1.1-Forces['Defender']['Units'][Key]['Roll']/10.
                         +DBonus/10.)
        DDist = convolve([DDist,next_roll])

    # Get outcomes for different results
    results={}
    for k in range(ADist.shape[0]):
        for l in range(ADist.shape[1]):
            for m in range(DDist.shape[0]):
                for n in range(DDist.shape[1]):
                    out = Resolve_Round(Forces, [n,l], STs=[k,m])
                    for outcome in out:
                        # Check for Yin racial ability
                        # Note: Dealing damage to Mercs with this 
                        #       is not implemented  
                        Keys=['Attacker','Defender']
                        for z, Key in enumerate(Keys):
                            Specials = [a[0] for a 
                                        in outcome[0][Key]['Special']] 
                            if 'Yin' in Specials:
                                Ship = 'None'
                                Ships = outcome[0][Key]['Units'].keys()
                                if 'DD' in [a[:2] for a in Ships]:
                                    Ship = 'DD%i'%max([int(a[2:]) for a in
                                                       Ships if a[:2] == 'DD'])
                                if 'CA' in [a[:2] for a in Ships]:
                                    Ship = 'CA%i'%max([int(a[2:]) for a in
                                                       Ships if a[:2] == 'CA'])
                                if Ship!='None':
                                    y = Specials.index('Yin')
                                    Targets = outcome[0][Key]['Special'][k][1:]
                                    del outcome[0][Key]['Special'][k]
                                    Success = False
                                    Key2 = Keys[1 - z]
                                    while ((not Success) and Targets != []):
                                        Target = Targets.pop(0)
                                        if (Target in outcome[0][
                                                Key2]['Units'].keys() and
                                            outcome[0][Key2]['Units'][
                                                Target]['HP'] == 1):
                                            Success = True
                                            del outcome[0][Key]['Units'][Ship]
                                            while (Ship in 
                                                   outcome[0][Key]['DmgOrder']):
                                                outcome[0][Key][
                                                    'DmgOrder'].remove(Ship)
                                            while (Ship in outcome[0][Key][
                                                    'RepairOrder']):
                                                outcome[0][Key][
                                                    'RepairOrder'].remove(Ship)
                                            del outcome[0][Key2]['Units'][
                                                Target]
                                            while (Target in outcome[
                                                    0][Key2]['DmgOrder']):
                                                outcome[0][Key2][
                                                    'DmgOrder'].remove(Target)
                                            while (Target in outcome[0][Key2][
                                                    'RepairOrder']):
                                                outcome[0][Key2][
                                                    'RepairOrder'].remove(
                                                    Target)
                                            if Target == 'FS_Nekro':
                                                outcome[0]['Attacker'][
                                                    'Units']={}
                                                outcome[0]['Attacker'][
                                                    'DmgOrder']=[]
                                                outcome[0]['Attacker'][
                                                    'RepairOrder']=[]
                                                outcome[0]['Defender'][
                                                    'Units']={}
                                                outcome[0]['Defender'][
                                                    'DmgOrder']=[]
                                                outcome[0]['Defender'][
                                                    'RepairOrder']=[]
                        BKey = BattleKey(outcome[0])
                        if BKey not in results.keys():
                            results[BKey] = (ADist[k,l] * DDist[m,n] 
                                             * outcome[1])
                        else:
                            results[BKey] += (ADist[k,l] * DDist[m,n] 
                                              * outcome[1])
 
    # Probability of no hits ad inifinitum
    nohits = 1
    if BattleKey(Forces) in results.keys():
        nohits /= (1-results[BattleKey(Forces)])

    out = {}
    for rKey in results.keys():
        if rKey!=BattleKey(Forces):
            p = results[rKey]
            result = InverseBattleKey(rKey)
            if (result['Attacker']['Units'].keys()!=[] and 
                result['Defender']['Units'].keys()!=[]):
                New=Battle(result,BattleDict=BattleDict)
                for BKey in New[0].keys():
                    if BKey not in out.keys():
                        out[BKey] = nohits * p * New[0][BKey]
                    else:
                        out[BKey] += nohits * p * New[0][BKey]
                FctCalls+=New[1]
                NewBattles=[a for a in New[2].keys() 
                            if a not in BattleDict.keys()]
                for Key in NewBattles:
                    BattleDict[Key]=NewBattles[Key]
            else:
                if rKey not in out.keys():
                    out[rKey] = nohits * p
                else:
                    out[rKey] += nohits * p
                    
    BattleDict[BattleKey(Forces)] = out 
    
    cost_res = ResCost(Forces,out)

    return out, FctCalls+1, BattleDict, cost_res

def NiceOut(Lines,Alignment):
    """Nicely aligned output.

    Arguments:
    Lines -- List containing a string for each column of output.
    Alignment -- String specifyin the alignment of each column, e.g 'llr'
    """
    #Determine required whitespace
    k=max([len(a) for a in Lines])
    ColLen=[[len(b) for b in a] for a in Lines]
    #ColLen=[]
    Widths=[max([a[l] for n,a in enumerate(ColLen) if len(a)>l])
            for l in range(k)]
    for l,line in enumerate(Lines):
        for m,column,align in zip(range(len(line)),line,list(Alignment)):
            if align=='l':
                n=Widths[m]-len(column)
                print column+' '.join(['' for a in range(n+1)]),
            else:
                n=Widths[m]-len(column)
                print ' '.join(['' for a in range(n+1)])+column,
        print

def ResValue(Forces):
    """
    Returns resource value of forces.
    """
    res_val = [sum([b['Cost'] for _, b in Forces['Attacker']['Units'].items()]),
               sum([b['Cost'] for _, b in Forces['Defender']['Units'].items()])]
    return np.array(res_val)

def ResCost(Forces, Outcome):
    total_res = ResValue(Forces)
    cost_res = np.zeros(2)
    for a,b in Outcome.items():
        cost_res += b * (total_res - ResValue(InverseBattleKey(a)))
    return cost_res

def PrepareForces(Ships_Att,Ships_Def):
    Dmg_Default = ['FT','DD','CA','CV','DN','WS']
    
    Forces={'Space':True}
    Forces['Attacker'] = {'Units':{}, 'Tech':[],'RepairOrder':[],'Rerolls':0}
    Forces['Defender'] = {'Units':{}, 'Tech':[],'RepairOrder':[],'Rerolls':0}
    Forces['Attacker']['Bonus'] = []
    Forces['Defender']['Bonus'] = []
    Forces['Attacker']['Special'] = []
    Forces['Defender']['Special'] = []
    
    Units_Att = {}
    for Type, Number in Ships_Att.items():
        for k in range(1,Number+1):
            Units_Att['{}{}'.format(Type,k)] = deepcopy(_Base_Values[Type])
    Forces['Attacker']['Units'] = Units_Att

    Dmg_Att = []
    MaxMaxHP = max([b['MaxHP'] for _, b in Units_Att.items()])
    for k in range(MaxMaxHP,0,-1):
        Ships = [a for a, b in Units_Att.items() if b['MaxHP'] >= k]
        Dmg_Att += sorted(Ships, key=lambda a: (Dmg_Default.index(a[:2]) * 1000
                                                - int(a[2:])))
    Forces['Attacker']['DmgOrder'] = Dmg_Att

    Units_Def = {}
    for Type, Number in Ships_Def.items():
        for k in range(1,Number+1):
            Units_Def['{}{}'.format(Type,k)] = deepcopy(_Base_Values[Type])
    Forces['Defender']['Units'] = Units_Def

    Dmg_Def = []
    MaxMaxHP = max([b['MaxHP'] for _, b in Units_Def.items()])
    for k in range(MaxMaxHP,0,-1):
        Ships = [a for a, b in Units_Def.items() if b['MaxHP'] >= k]
        Dmg_Def += sorted(Ships, key=lambda a: (Dmg_Default.index(a[:2]) * 1000
                                                - int(a[2:])))
    Forces['Defender']['DmgOrder'] = Dmg_Def

    return Forces

def FullBattle(Ships,BattleDict=None,Tech=None):
    """
    Ships = (Ships_Att, Ships_Def)
    """
    Battles = [(Ships,1)]
    if Tech is None:
        Tech = ([],[])
    
    # Attacker AFB
    for k in range(2):
        if 'DD' in Ships[k].keys() and 'FT' in Ships[1-k].keys():
            if Ships[k]['DD'] > 0 and Ships[1-k]['FT'] > 0:
                if 'ATD' in Tech[k]:
                    n_dice = 3 * Ships[k]['DD']
                    roll = _Base_Values['DD']['Roll'] + 2
                else:
                    n_dice = 2 * Ships[k]['DD']
                    roll = _Base_Values['DD']['Roll']

                prob = dist(n_dice,1.1-roll/10.)[0]
                
                if n_dice > Ships[1-k]['FT']:
                    prob = np.concatenate((prob[:Ships[1-k]['FT']],
                                           np.array([np.sum(prob[Ships[1-k]['FT']:])])))

                NewBattles = []
                for B in Battles:
                    for hits,p in enumerate(prob):
                        NewBattle = (deepcopy(B[0]),B[1]*p)
                        NewBattle[0][1-k]['FT'] -= hits
                        NewBattles.append(NewBattle)
                Battles = deepcopy(NewBattles)

    out = {}
    for B in Battles:
        B_out = Battle(PrepareForces(*B[0]),BattleDict=BattleDict)
        BattleDict = B_out[2]
        for key, p in B_out[0].items():
            if key in out.keys():
                out[key] += B[1] * p
            else:
                out[key] = B[1] * p

    return out, BattleDict

def EasyOutcomes(a):
    out = []
    out += [sum([a[b] for b in a.keys() 
                 if (InverseBattleKey(b)['Defender']['Units']=={} 
                 and InverseBattleKey(b)['Attacker']['Units']!={})])]
    out += [sum([a[b] for b in a.keys() 
                 if (InverseBattleKey(b)['Defender']['Units']=={} 
                 and InverseBattleKey(b)['Attacker']['Units']=={})])]
    out += [sum([a[b] for b in a.keys() 
                 if (InverseBattleKey(b)['Defender']['Units']!={} 
                 and InverseBattleKey(b)['Attacker']['Units']=={})])]
    return out

def check_fleet(units,res,fs,count):
    if sum(units.values()) > count:
        return False, 'Count'
    elif sum([n for key, n in units.items() if key != 'FT']) > fs:
        return False, 'FS'
    elif sum([n*_unit_cost[key] for key, n in units.items()]) > res:
        return False, 'Resources'
    elif units['FT'] > 6 * units['CV']:
        return False, 'Capacity'
    else:
        return True, 'OK'

def design_fleet(res,fs,count,no_CV_wo_FT=True):
    units = [{'WS': 0, 'DN': 0, 'CA': 0, 'DD': 0, 'CV': 0, 'FT': 0}]
    
    for unit_type in ['WS','DN','CV','CA','DD']:
        new_units = []
        for k in range(0,1 + min(int(res/_unit_cost[unit_type]),
                                 fs,
                                 count,
                                 _unit_available[unit_type])):
            for unit in units:
                new = deepcopy(unit)
                new[unit_type] = k
                if check_fleet(new,res,fs,count)[0]:
                    new_units.append(new)
        units = deepcopy(new_units)
    
    for unit in units:
        FTs = min(count - sum(unit.values()),
                  int((res - sum([n*_unit_cost[key] for key, n in unit.items()])) 
                      / _unit_cost['FT']),
                  6 * unit['CV'])
        FTs = (FTs / 2) * 2
        unit['FT'] = FTs
        
    if no_CV_wo_FT:    
        new_units = []
        for unit in units:
            OK = True
            if unit['CV'] > (unit['FT'] - 1) / 6 + 1:
                OK = False
            if OK:
                new_units.append(deepcopy(unit))
        units = deepcopy(new_units)
                
    new_units = []
    for k,unit1 in enumerate(units):
        included = False
        for unit2 in units[:k]+units[k+1:]:
            if False not in [(unit1[key] <= unit2[key]) 
                             or (unit1[key] == 0 and unit2[key] == 0) 
                             for key in unit1.keys()]:
                included = True
                break
        if not included:
            new_units.append(deepcopy(unit1))
    
    units = deepcopy(new_units)
    for unit_type in ['FT','DD','CA','CV','DN','WS']:
        units = sorted(units, key=lambda x: x[unit_type])
    
    units = units[::-1]

    return units

def ultimate_fleet(res,fs,count,no_CV_wo_FT=True,BattleDict=None):
    Fleets =  design_fleet(res,fs,count,
                           no_CV_wo_FT=no_CV_wo_FT)

    out = {}
    for k, Fleet1 in enumerate(Fleets[:-1]):
        for l, Fleet2 in enumerate(Fleets[k+1:]):
            B_out = FullBattle((Fleet1,Fleet2),BattleDict=BattleDict)
            BattleDict = B_out[1]
            out[(k,l+k+1)] = B_out[0]
    
    return out, Fleets, BattleDict

## def print_Battle(ANum, ARoll, DNum, DRoll,Dict=False,BattleDict={}):
##     a,b,c,d=Battle(ANum, ARoll, DNum, DRoll,BattleDict=BattleDict)
##     print '%i@%i vs. %i@%i'%(ANum,ARoll,DNum,DRoll)
##     print 'Function Calls: %i'%(c)
##     Out=[['Attacker wins:','%.2f%%'%(sum(a[1:])*100)],
##          ['Tie:','%.2f%%'%(a[0]*100)],
##          ['Defender wins:','%.2f%%'%(sum(b[1:])*100)]]
##     NiceOut(Out,'lr')
##     print 'Remaining GFs:'
##     Out=[['N','Attacker','Defender']]
##     Out+=[['0','%.2f%%'%((a[0]+sum(b[1:]))*100),
##            '%.2f%%'%((b[0]+sum(a[1:]))*100)]]
##     for k in range(1,max(ANum+1,DNum+1)):
##         if k<=min(ANum,DNum):
##             Out+=[[str(k),'%.2f%%'%(a[k]*100),'%.2f%%'%(b[k]*100)]]
##         else:
##             if ANum>DNum:
##                 Out+=[[str(k),'%.2f%%'%(a[k]*100),'-']]
##             else:
##                 Out+=[[str(k),'-','%.2f%%'%(b[k]*100)]]
##     NiceOut(Out,'rrr')
##     if Dict: return d

## if __name__ == '__main__':
##     try:
##         ANum=int(sys.argv[1])
##         ARoll=int(sys.argv[2])
##         DNum=int(sys.argv[3])
##         DRoll=int(sys.argv[4])
##     except:
##         print 'Usage: python '+sys.argv[0]+' ANum ARoll DNum DRoll'
##         sys.exit()

##     BattleDict={}
##     if '-pi' in sys.argv:
##         f=file(sys.argv[sys.argv.index('-pi')+1],'r')
##         BattleDict=Pickle.load(f)
##         f.close()

##     NewDict=print_Battle(ANum,ARoll,DNum,DRoll,Dict=True,BattleDict=BattleDict)

##     if '-po' in sys.argv:
##         f=file(sys.argv[sys.argv.index('-po')+1],'w')
##         Pickle.dump(NewDict,f)
##         f.close()
