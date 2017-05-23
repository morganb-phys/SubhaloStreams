import numpy as np
import matplotlib.pyplot as plt

class Stream:
    
    G = 6.67408e-11
    nump = 10
    psi0 = np.linspace(-5,5,nump)
    
    def __init__(self,mass,rstream,rsub,impact,strvel,subvel,t):
        self.m = mass   # Mass of the subhalo
        self.r0 = rstream   # Radius of the circular orbit of the stream
        self.rs = rsub   # Radius of the subhalo
        self.b = impact   # Impact parameter for the subhalo and stream
        self.v = strvel    # Velocity of the stream
        self.wvec = subvel    # Relative velocity of the halo
        self.t = t    # Time elapsed after the halo impacted the stream

        self.wperp = np.sqrt(subvel[0]**2 + subvel[2]**2)
        self.wpar = v[1] - wvec[1]
        self.w = np.sqrt(subvel[0]**2 + subvel[2]**2 + (v[1] - wvec[1])**2)
        
        self.calc_dv()
        self.calc_psi_rho()
        self.calc_dx()
        self.calc_dxdot()


    '''
    These are the functions which calculate variables that will be used in the other
    functions, but which do not calculate attributes of the object
    '''

    def d2rphi(self):
        r0 = self.r0
        m = self.m
        rs = self.rs

        phideriv = -self.G*m*((2.*rs+3.*r0)/(r0**2*(rs+r0)**2)-2.*np.log(1.+r0/rs)/r0**3)

        return phideriv
        
    def gamma(self):
        g = np.sqrt(3.+self.r0**2/self.v[1]**2*self.d2rphi())
        return g

    def gT(self):
        angle = self.gamma()*self.v[1]*self.t/self.r0
        return angle


    '''
    These are the functions which set attributes of the object
    '''

    def calc_dv(self):
        M = self.m
        r0 = self.r0
        wperp = self.wperp
        wpar= self.wpar
        w = self.w
        wvec = self.wvec
        b = self.b
        rs = self.rs
        G = self.G
        psi0 = self.psi0
        
        deltav = np.zeros([3,self.nump])
        deltav[0] = 2.*G*M/r0**2/wperp**2/w*(b*w**2*wvec[2]/wperp-psi0*r0*wpar*wvec[0])/(psi0**2+(b**2+rs**2)*w**2/r0**2/wperp**2)
        deltav[1] = -2.*G*M*psi0/r0/w/(psi0**2+(b**2+rs**2)*w**2/r0**2/wperp**2)
        deltav[2] = -2.*G*M/r0**2/wperp**2/w*(b*w**2*wvec[0]/wperp-psi0*r0*wpar*wvec[2])/(psi0**2+(b**2+rs**2)*w**2/r0**2/wperp**2)

        self.dv = deltav
        

    def calc_psi_rho(self):
        gam = self.gamma()
        gT = self.gT()

        t = self.t
        r0 = self.r0
        v = self.v
        wperp  = self.wperp
        wpar = self.wpar
        wvec = self.wvec
        w = self.w
        b = self.b
        rs = self.rs
        psi0 = self.psi0

        tau = w*r0**2/2./self.G/self.m
        
        f = (4.-gam**2)/gam**2*t/tau - 4.*np.sin(gT)/gam**3*r0/v[1]/tau+2.*(1.-np.cos(gT))/gam**2*wperp*wvec[0]/wpar**2*r0/v[1]/tau
        g = 2.*(1-np.cos(gT))*b*w**2*wvec[2]*r0/(gam**2*r0*wperp**3*v[1]*tau)
        B2 = (b**2+rs**2)*w**2/(r0**2*wperp**2)

        print f
        print g
        print B2
        
        self.psi = psi0+(f*psi0-g)/(psi0**2+B2)
        self.rho = (1+f/(psi0**2+B2)-(f*psi0-g)*2.*psi0/(psi0**2+B2)**2)**(-1)


    def calc_dx(self):
        r0 = self.r0
        v = self.v
        dv = self.dv

        gT = self.gT()
        
        deltax = np.zeros([3,self.nump])
        deltax[0] = 2.*r0*dv[1]/v[1]*(1.-np.cos(gT))/self.gamma()**2
        deltax[2] = dv[2]/v[1]*np.sin(v[1]*self.t/r0)
        
        self.dx = deltax

    def calc_dxdot(self):
        v = self.v
        dv = self.dv
        gamma = self.gamma()
        gT = self.gT()
        
        deltaxdot = np.zeros([3,self.nump])
        deltaxdot[0] = 2.*dv[1]/gamma*np.sin(gT)+dv[0]*np.cos(gT)
        deltaxdot[1] = -dv[1]*(2.-gamma**2)/gamma**2+2.*v[1]*np.cos(gT)/gamma**2-dv[0]*np.sin(gT)/gamma
        deltaxdot[2] = dv[2]*np.cos(v[1]*self.t/self.r0)

        self.dxdot = deltaxdot

    

if __name__ == "__main__":


    M = 10.**7
    r0 = 3.08567758e16
    b = 0.
    v = [0.,200.,0.]
    wvec = [100.,0.,0.]
    rs = 0.625*r0
    time = 450.e6

    stream1 = Stream(M,r0,rs,b,v,wvec,time)

    print stream1.rho
    dx = stream1.dx
    psi = stream1.psi

    plt.figure()

    plt.plot(psi,dx[0])
    plt.plot(psi,dx[2])
    plt.show()
    
