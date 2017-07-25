import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import simps
import matplotlib.pyplot as plt
import StreamModel as sm


def fitData(stream,guess):

    #Generate the data that you wnat to fit to
    dxstd = [10.,0.,0.02*np.pi/180.]
    dxdotstd = [0.3,0.15,0.1]

    datapsi1,datadx1,datadxdot1 = sm.GenData(stream,100,[dxstd,dxdotstd])

    #Find the mass that minimizes the log likelihood function
    fit = minimize(logLikelihood,guess,args=([datapsi1,datadx1[0]],dxstd[0]))
    fit2 = minimize(lambda x: logLikelihood(x,[datapsi1,datadx1[0]],dxstd[0]),guess)

    print fit.x, fit2.x

def logLikelihood(guess,data,std):

    mass = guess[0]
    psi = data[0]
    dx = data[1]

    print mass

    #Create a stream with the current mass
    stream = sm.Stream(mass,9800.,625.,0.,[0.,0.,150.],450.)

    #Find the density at the point 
    f = interp1d(stream.psi,stream.rho)
    density = f(psi)

    g = interp1d(stream.psi,stream.dx[0])
    dxmodel = g(psi)

    logL = logGaussian(data,dxmodel,std) + np.log(density)
    neglogL = -np.sum(logL)
    
    return neglogL

def logGaussian(data,model,std):

    loggauss = -(0.5*np.log(2.*np.pi*std**2)+(data-model)**2*0.5/std**2)
    return loggauss


def prob(data):

    L = len(data[0])
    psidata = data[0]
    dxdata = data[1]
    mass = np.logspace(5,10,L)
    std = 10.
    p = np.zeros(L)
    
    for i in range(100):
        stream = sm.Stream(mass[i],9800.,625.,0.,[0.,0.,150.],450.)

        psi = stream.psi
        rho = stream.rho/simps(stream.rho,psi)
        dx = stream.dx[0]

        rhofunc = interp1d(psi,rho)
        dxfunc = interp1d(psi,dx)
    
        p[i] = np.prod(1./np.sqrt(2.*np.pi*std**2)*np.exp(-(dxdata-dxfunc(psidata))**2/2./std**2)*rhofunc(psidata))

    plt.figure()
    plt.semilogx(mass,p)
    plt.show()

                   
stream1 = sm.Stream(1.e8,9800.,625.,0.,[0.,0.,150.],450.)

dxstd = [10.,0.,0.02*np.pi/180.]
dxdotstd = [0.3,0.15,0.1]
datapsi1,datadx1,datadxdot1 = sm.GenData(stream1,100,[dxstd,dxdotstd])

prob([datapsi1,datadx1[0]])

fit = fitData(stream1,9.99e7)


