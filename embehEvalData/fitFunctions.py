from numpy import *
from scipy.special import erf, erfc

print('fitFunctions sucessfully imported:')


def Heaviside(x):
    output = np.ones_like(x)
    output[x < 0] = 0
    output[x >= 0] = 1
    return output

def gauss(x, sigma):
    #normalised to area one
    return 1/sqrt(2*pi)/sigma*exp(-x**2/(2*sigma**2))
    
def gaussHeight(x, sigma, A, offset):
    #normalised to height one with the ability to offset aswell
    return A*exp(-x**2/(2*sigma**2))+offset
    
def expDecay(x, tau, A):
    return A*Heaviside(x)*(exp(-x/tau)-1) + 1
    
def SglExpDecay(x, t0, tau, A, B):
    return A-(1-B*exp(-(x-t0)/tau))*Heaviside(t0-x)
    
def expDecayOffs(x, tau, A, y0):
    return A*Heaviside(x)*(exp(-x/tau)-1) + 1 + y0
    
def doubleExpDecay(x, tau1, tau2, A, B):
    return A*Heaviside(x)*(exp(-x/tau1)-1)+B*Heaviside(x)*(exp(-x/tau2)-1) + 1

def doubleExpDecay_II(x, tau1, tau2, A, B, t0, offset):
    return A*Heaviside(x-t0)*(exp(-(x-t0)/tau1)-1)+B*Heaviside(x-t0)*(exp(-(x-t0)/tau2)-1) + offset
    
def expConvGauss(x, tau, A, sigma):
    term1 = exp(-x/tau)*exp(sigma**2/(2*tau**2))*(erf((sigma**2-x*tau)/(sqrt(2)*sigma*tau))-1)
    return -A/2*(term1)

def expConvGaussNormalised(x, tau, sig):
    exp_enu = sig**2-2*(x)*tau
    exp_den = 2*tau**2
    erf_enu = sig**2+(-x)*tau
    erf_den = sqrt(2)*sig*tau
    return 1/2 * exp(exp_enu/exp_den) * (erf(erf_enu/erf_den)-1)

def expConvGauss2(x, t0, tau, A, sigma):
    term1 = exp(-(x-t0)/tau)*exp(sigma**2/(2*tau**2))*(erf((sigma**2-(x-t0)*tau)/(sqrt(2)*sigma*tau))-1)
    return -A/2*(term1)
    
def AHConvGauss(x, A, sigma):
    term1 = erf(x/(sqrt(2)*sigma))
    return -A/2*(term1-2/A+1)
    
def ABHConvGauss(x, A, B, sigma):
    term1 = -A/2*erf(x/(sqrt(2)*sigma))
    term2 = -B/2*erf(x/(sqrt(2)*sigma))
    return term1+term2+1-A/2-B/2

def expConvGaussApprox(x, tau, A, sigma):
    C = sqrt(2/pi)*exp(-x**2/(2*sigma**2))
    r = sigma/tau
    k = C/r - x*C/(sigma*r**2) + C*(sigma**2-x**2)/(sigma**2*r**3)#+C*x*(3*sigma**2-x**2)/(sigma**3*r**4)-C*(3*sigma**4-6*sigma**2*x**2+x**4)/(sigma**4*r**5)
    return -A/2*k

def doubleDecay(x, t0, tau1, tau2, A, q):
    B = (1-q)*A    
    A = q*A
    return doubleExpDecay(x-t0,tau1,tau2,A,B)
    
def doubleDecay2(x, t0, tau1, tau2, A, B):
    return doubleExpDecay(x-t0,tau1,tau2,A,B)

def doubleDecaySingleConv(x, t0, tau1, tau2, A, B, sig):
    C = -(A+B)
    return -A*expConvGaussNormalised(x-t0, tau1, sig) -B*expConvGaussNormalised(x-t0, tau2, sig) +1/2*C*erfc(-(x-t0)/(sqrt(2*sig**2))) + 1

def doubleDecayDoubleConv(x, t0, tau1, tau2, A, q, alpha, sigS, sigH):

	# A = overall amplitude
	# q = tau1 fraction of A
	# alpha = slicing fraction

    B = (1-q)*A    
    A = q*A

    term1 = expConvGauss(x-t0,tau1,A,sigS)
    term2 = expConvGauss(x-t0,tau2,B,sigS)
    term3 = ABHConvGauss(x-t0,A,B,sigS)
    if sigH/tau1 < 5:
        term4 = expConvGauss(x-t0,tau1,A,sigH)
    else:
        term4 = expConvGaussApprox(x-t0,tau1,A,sigH)

    if sigH/tau2 < 5:
        term5 = expConvGauss(x-t0,tau2,B,sigH)
    else:
        term5 = expConvGaussApprox(x-t0,tau2,B,sigH)
    
    term6 = ABHConvGauss(x-t0,A,B,sigH)

    return alpha*(term1 + term2 + term3)+(1-alpha)*(term4 + term5 + term6)
    
def doubleDecayDoubleConv2(x, t0, tau1, tau2, A, B, alpha, sigS, sigH):
    
    term1 = expConvGauss(x-t0,tau1,A,sigS)
    term2 = expConvGauss(x-t0,tau2,B,sigS)
    term3 = ABHConvGauss(x-t0,A,B,sigS)
    
    if alpha == 1:
        return term1 + term2 + term3
        
    else:
        
        if sigH/tau1 < 5:
            term4 = expConvGauss(x-t0,tau1,A,sigH)
        else:
            term4 = expConvGaussApprox(x-t0,tau1,A,sigH)

        if sigH/tau2 < 5:
            term5 = expConvGauss(x-t0,tau2,B,sigH)
        else:
            term5 = expConvGaussApprox(x-t0,tau2,B,sigH)
        
        term6 = ABHConvGauss(x-t0,A,B,sigH)

        return alpha*(term1 + term2 + term3)+(1-alpha)*(term4 + term5 + term6)

def doubleDecayConvScale(x, t0, tau1, tau2, A, q, alpha, sigS, sigH, I0):
    B = (1-q)*A    
    A = q*A

    term1 = expConvGauss(x-t0,tau1,A,sigS)
    term2 = expConvGauss(x-t0,tau2,B,sigS)
    term3 = ABHConvGauss(x-t0,A,B,sigS)
    if sigH/tau1 < 5:
        term4 = expConvGauss(x-t0,tau1,A,sigH)
    else:
        term4 = expConvGaussApprox(x-t0,tau1,A,sigH)

    if sigH/tau2 < 5:
        term5 = expConvGauss(x-t0,tau2,B,sigH)
    else:
        term5 = expConvGaussApprox(x-t0,tau2,B,sigH)
    
    term6 = ABHConvGauss(x-t0,A,B,sigH)

    return I0*(alpha*(term1 + term2 + term3)+(1-alpha)*(term4 + term5 + term6))
    
def DecayConv(x, t0, tau, A, sigma):
    
    term1 = expConvGauss(x-t0,tau,A,sigma)
  
    term3 = AHConvGauss(x-t0,A,sigma)
    
    return term1 + term3

def doubleDecayConvSqrd(x, t0, tau1, tau2, A, q, alpha, sigS, sigH):
    B = (1-q)*A    
    A = q*A

    term1 = expConvGauss(x-t0,tau1,A,sigS)
    term2 = expConvGauss(x-t0,tau2,B,sigS)
    term3 = ABHConvGauss(x-t0,A,B,sigS)
    if sigH/tau1 < 5:
        term4 = expConvGauss(x-t0,tau1,A,sigH)
    else:
        term4 = expConvGaussApprox(x-t0,tau1,A,sigH)

    if sigH/tau2 < 5:
        term5 = expConvGauss(x-t0,tau2,B,sigH)
    else:
        term5 = expConvGaussApprox(x-t0,tau2,B,sigH)
    
    term6 = ABHConvGauss(x-t0,A,B,sigH)

    return (alpha*(term1 + term2 + term3)+(1-alpha)*(term4 + term5 + term6))**2
    
def fitReflectivity(x, mu, A, const):
    model = A*(x-mu)**(-4) + const
    return model

def pseudoVoigt(x, mu, sigma, A, alpha):
    model = A*(alpha * 1/(1 + ((x-mu)/sigma)**2) + (1-alpha)*exp(-log(2)*((x-mu)/sigma)**2))
    return model

def reflectivityPseudoVoigt(x, ampl1, const1, center1, ampl2, center2, sigma2, alpha2):
    model1 = ampl1*(x-center1)**(-4) + const1
    model2 = ampl2*(alpha2 * 1/(1 + ((x-center2)/sigma2)**2) + (1-alpha2)*exp(-log(2)*((x-center2)/sigma2)**2))
    return model1 + model2 