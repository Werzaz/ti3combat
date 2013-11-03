import numpy as np
import matplotlib.pyplot as P

_d_outer=0.2 # Probability of survival of single FT against outer defenses
_d_inner=0.1 # Probability of survival of single FT against inner defenses

def binom(n,k): 
    return np.prod([float(n-k+i)/i for i in range(1,k+1)])

def p_outer(n,k,d_outer=_d_outer): 
    return binom(n,k)*(1-d_outer)**(n-k)*(d_outer)**k

def p_inner(n,d_inner=_d_inner): 
    return 1-(1-d_inner)**n

def p_total(n,d_inner=_d_inner,d_outer=_d_outer): 
    return np.sum([p_outer(n,k,d_outer=d_outer)*p_inner(k,d_inner=d_inner) 
                   for k in range(1,n+1)])

def p_survives(n,k,d_inner=_d_inner): 
    if k > 0: return (1-d_inner)**(n-k)*d_inner
    else: return (1-d_inner)**n

def p_survivors(n,k,d_inner=_d_inner,d_outer=_d_outer): 
    return np.sum([(p_outer(n,m,d_outer=d_outer) 
                    * p_survives(m,k,d_inner=d_inner)) for m in range(k,n+1)])

def n_survivors(n,d_inner=_d_inner,d_outer=_d_outer):
    return np.sum([k*p_survivors(n,k,d_inner=d_inner,d_outer=d_outer) 
                   for k in range(1,n+1)])

def std_survivors(n,d_inner=_d_inner,d_outer=_d_outer):
    out = np.sum([k**2*p_survivors(n,k,d_inner=d_inner,d_outer=d_outer) 
                  for k in range(1,n+1)])
    out -= n_survivors(n,d_inner=d_inner,d_outer=d_outer)**2
    return np.sqrt(out)

if __name__ == '__main__':
    props=[p_total(n) for n in range(1,101)]
    P.plot(range(1,101),props)
    P.xlim((1,100))
    P.ylim((0,1))
    P.xlabel('Number of FTs', fontsize=16)
    P.ylabel('Probability of succes', fontsize=16)
    P.grid(True)
    P.savefig('p_total.png')

    P.clf()
    n_surv=np.array([n_survivors(n) for n in range(1,101)])
    std_surv=np.array([std_survivors(n) for n in range(1,101)])
    P.fill_between(range(1,101),n_surv+std_surv,n_surv-std_surv,color='c',
                   alpha=0.5,edgecolor='none')
    P.plot(range(1,101),n_surv+std_surv,'b--')
    P.plot(range(1,101),n_surv-std_surv,'b--')
    P.plot(range(1,101),n_surv,'b-')
    P.xlim((1,100))
    P.ylim((0,20))
    P.xlabel('Number of FTs', fontsize=16)
    P.ylabel('Expected number of surviving FTs', fontsize=16)
    P.grid(True)
    P.savefig('n_survivors.png')


