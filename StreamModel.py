import numpy as np
import matplotlib.pyplot as plt
from galpy.potential import NFWPotential


class Stream:

    nfwp= NFWPotential(normalize=1.,a=14./10.)
    G = 4.302e-3 #pc*M_sun^-1*(km/s)^2
    nump = 50
    psi0 = np.linspace(-20,20,nump)*np.pi/180.
    
    def __init__(self,mass,rstream,rsub,impact,subvel,t):
        self.m = mass   # Mass of the subhalo
        self.r0 = rstream   # Radius of the circular orbit of the stream
        self.rs = rsub   # Radius of the subhalo
        self.b = impact   # Impact parameter for the subhalo and stream
        self.wvec = subvel    # Relative velocity of the halo
        self.t = t    # Time elapsed after the halo impacted the stream
        self.vy = self.nfwp.vcirc(self.r0/10000.)*168.2 # in km/s

        # Calculate the different components of the subhalo relative velocity
        self.wperp = np.sqrt(subvel[0]**2 + subvel[2]**2)
        self.wpar = self.vy - wvec[1]
        self.w = np.sqrt(subvel[0]**2 + subvel[2]**2 + (self.vy - wvec[1])**2)

        print self.w

        self.calc_gamma()
        self.calc_dv()      # Calculates the change in initial velocity as a function of angle       
        self.calc_psi_rho()     # Calculates the change in angle as a function of time and the change in density
        self.calc_dx()      # Calculates the perturbations in the x and z direction of the stream
        self.calc_dxdot()   # Calculates the change in the stream velocity as a function of time and angle


    '''
    These are the functions which calculate variables that will be used in the other
    functions, but which do not calculate attributes of the object
    '''
        
    def gT(self):
        angle = self.gamma*self.vy*self.t/self.r0
        return angle

    '''
    These are the functions which set attributes of the object
    '''

    def calc_gamma(self):
        g = 3. + (self.r0)**2*self.nfwp.R2deriv(self.r0,0.)/self.vy**2.
        print 'potential: ',self.nfwp.R2deriv(0.98,0.)
        print 'r02*pot: ', (self.r0)**2*self.nfwp.R2deriv(self.r0/10000.,0.)
        self.gamma = np.sqrt(g)


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
        deltav[2] = -2.*G*M/r0**2/wperp**2/w*(b*w**2*wvec[0]/wperp+psi0*r0*wpar*wvec[2])/(psi0**2+(b**2+rs**2)*w**2/r0**2/wperp**2)

        self.dv = deltav
        

    def calc_psi_rho(self):
        gam = self.gamma
        gT = self.gT()

        t = self.t
        r0 = self.r0
        vy = self.vy
        wperp  = self.wperp
        wpar = self.wpar
        wvec = self.wvec
        w = self.w
        b = self.b
        rs = self.rs
        psi0 = self.psi0

        tau = w*r0**2/2./self.G/self.m
        
        f = (4.-gam**2)/gam**2*t/tau - 4.*np.sin(gT)/gam**3*r0/vy/tau+2.*(1.-np.cos(gT))/gam**2*wperp*wvec[0]/wpar**2*r0/vy/tau
        g = 2.*(1-np.cos(gT))*b*w**2*wvec[2]*r0/(gam**2*r0*wperp**3*vy*tau)
        B2 = (b**2+rs**2)*w**2/(r0**2*wperp**2)
        
        self.psi = psi0+(f*psi0-g)/(psi0**2+B2)
        self.rho = (1.+(f*B2-f*psi0**2+2.*g*psi0)/(psi0**2+B2)**2)**(-1)

    def calc_dx(self):
        r0 = self.r0
        vy = self.vy
        dv = self.dv

        gT = self.gT()
        
        deltax = np.zeros([3,self.nump])
        deltax[0] = 2.*r0*dv[1]/vy*(1.-np.cos(gT))/self.gamma**2+r0*dv[0]/vy*np.sin(gT)/self.gamma
        deltax[2] = dv[2]/vy*np.sin(vy*self.t/r0)
        
        self.dx = deltax

    def calc_dxdot(self):
        vy = self.vy
        dv = self.dv
        gamma = self.gamma
        gT = self.gT()
        
        deltaxdot = np.zeros([3,self.nump])
        deltaxdot[0] = 2.*dv[1]/gamma*np.sin(gT)+dv[0]*np.cos(gT)
        deltaxdot[1] = -dv[1]*(2.-gamma**2)/gamma**2+2.*dv[1]*np.cos(gT)/gamma**2-dv[0]*np.sin(gT)/gamma
        deltaxdot[2] = dv[2]*np.cos(vy*self.t/self.r0)

        print 'First: ', 2.*dv[1]/gamma*np.sin(gT)

        self.dxdot = deltaxdot

#def GenData(x,y,dy)
#    data = [(numpy.random.normal(y[i],dy[i]) for i in range(len(y))]


if __name__ == "__main__":


    m = 1.e8
    r0 = 9800.
    b = 0.
    wvec = [0.,0.,150.]
    rs = 625.
    time = 450.*1.02 # Myr*1.02(pc/(km/s)/Myr)

    stream1 = Stream(m,r0,rs,b,wvec,time)
    stream2 = Stream(1.e7,r0,250.,b,wvec,time)

    dx = stream1.dx
    psi = stream1.psi*180./np.pi
    dv = stream1.dv

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex='col', sharey='row')
    ax1.plot(psi,dx[0]/1000.)
    ax1.set_xlim([-20,20])
    ax1.set_ylim([-0.1,0.1])
    ax1.set_ylabel(r'$\Delta x$ (kpc)')
    ax2.plot(psi,stream1.dx[2]*180./np.pi)
    ax2.set_ylim([-1,1])
    ax2.set_ylabel(r'$\delta_z$')
    ax3.plot(psi,stream1.dxdot[0])
    ax3.set_ylim([-3,3])
    ax3.set_ylabel(r'$\Delta \dot x$')
    ax4.plot(psi,stream1.dxdot[1])
    ax4.set_ylim([-3,3])
    ax4.set_ylabel(r'$\Delta \dot y$')
    ax5.plot(psi,stream1.dxdot[2])
    ax5.set_ylim([-3,3])
    ax5.set_ylabel(r'$\Delta \dot z$')
    ax6.plot(psi,stream1.rho)
    ax6.set_ylabel(r'$\rho$')

    plt.figure()
    plt.plot(stream1.psi0*stream1.r0,stream1.dv[0],label='x')
    plt.plot(stream1.psi0*stream1.r0,stream1.dv[1],label='y')
    plt.plot(stream1.psi0*stream1.r0,stream1.dv[2],label='z')
    plt.ylabel(r'$\Delta v$')
    plt.xlabel(r'$\psi (^\circ)$')
    plt.legend()

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex='col', sharey='row')
    ax1.plot(psi,dx[0]/1000.)
    ax1.set_xlim([-20,20])
    ax1.set_ylim([-0.05,0.05])
    ax1.set_ylabel(r'$\Delta x$ (kpc)')
    ax2.plot(psi,stream2.dx[2]*180./np.pi)
    ax2.set_ylim([-0.5,0.5])
    ax2.set_ylabel(r'$\delta_z$')
    ax3.plot(psi,stream2.dxdot[0])
    ax3.set_ylim([-1,1])
    ax3.set_ylabel(r'$\Delta \dot x$')
    ax4.plot(psi,stream2.dxdot[1])
    ax4.set_ylim([-1,1])
    ax4.set_ylabel(r'$\Delta \dot y$')
    ax5.plot(psi,stream2.dxdot[2])
    ax5.set_ylim([-1,1])
    ax5.set_ylabel(r'$\Delta \dot z$')
    ax6.plot(psi,stream2.rho)
#    ax6.set_ylim([0.,0.05])
    ax6.set_ylabel(r'$\rho$')

    plt.figure()
    plt.plot(stream1.psi0*180./np.pi,dx[0]/1000.)
    plt.plot(stream1.psi*180./np.pi,dx[0]/1000.)
    
    plt.show()
