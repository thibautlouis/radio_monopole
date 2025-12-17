import numpy as np
import pylab as plt
import os

labelsize = 14
fontsize = 20

ref_freq_radio_GHz = 148
tucci_file_name = f"ns_148GHz_modC2Ex.dat"

def read_tucci_source_distrib(plot_fname=None):
    """
    Read the source distribution from tucci et al (https://arxiv.org/pdf/1103.5707.pdf) with flux (S) at ref frequency 148 GHz
    and optionnaly plot it. Should replicate the C2Ex model (bottom long dashed line of Fig14)
    """

    tucci = np.loadtxt(tucci_file_name)
    S = tucci[:, 0]
    dNdSdOmega = tucci[:, 1]
    
    if plot_fname is not None:
        plt.figure(figsize=(8,6))
        plt.loglog()
        plt.ylim(0.6, 100)
        plt.xlim(0.001, 20)
        plt.plot(S, dNdSdOmega * S ** (5/2), "--", color="red")
        plt.xlabel("S (Flux [Jy])", fontsize=fontsize)
        plt.tick_params(labelsize=labelsize)
        plt.ylabel(r"$S^{5/2}\frac{dN}{dS d\Omega} [Jy^{3/2} sr^{-1}]$", fontsize=fontsize)
        plt.savefig(plot_fname, bbox_inches="tight")
        plt.clf()
        plt.close()
        
    return S, dNdSdOmega
    
def get_mean_number_of_source_over_4pi(S, dNdSdOmega, plot_fname=None):

    """
    Get the mean number of sources in the full sky
    
    Parameters
    ----------
    S : 1d array
        source flux at the ref frequency in Jy
    dNdSdOmega : 1d array
        source distribution dN/(dS dOmega) corresponding to S
    plot_fname : str
        if not None save the plot in plot_fname
    """

    dS = np.gradient(S)
    mean_numbers_per_patch = dNdSdOmega * 4*np.pi * dS
    
    if plot_fname is not None:
        plt.figure(figsize=(8,6))
        plt.loglog()
        plt.plot(S, mean_numbers_per_patch)
        plt.xlabel(r"$S$ (Flux [Jy])", fontsize=fontsize)
        plt.ylabel(r"$N_{source}$", fontsize=fontsize)
        plt.tick_params(labelsize=labelsize)

            
        plt.savefig(plot_fname, bbox_inches="tight")
        plt.clf()
        plt.close()

    return mean_numbers_per_patch


def get_cumulative_monopole(S, dNdSdOmega, plot_fname=None):
    """
    get the monopole as a function of S_max
    """
    dS = np.gradient(S)
    monopole =   np.cumsum(dS * S  * dNdSdOmega)
    
    if plot_fname is not None:
        #plot the integrand
        plt.figure(figsize=(8,6))
        plt.loglog()
        plt.plot(S, S  * dS * dNdSdOmega)
        plt.xlabel(r"$S$ (Flux [Jy])", fontsize=fontsize)
        plt.ylabel(r"$S \left. \frac{dN}{dSd\Omega}\right|_{148 {\rm GHz}} dS [Jy . sr^{-1}]$", fontsize=fontsize)
        plt.tick_params(labelsize=labelsize)
        plt.savefig(plot_fname, bbox_inches="tight")
        plt.clf()
        plt.close()

    return monopole



os.makedirs("figs", exist_ok=True)

S, dNdSdOmega = read_tucci_source_distrib(plot_fname=f"figs/source_distrib_radio.png")
mean_numbers_per_patch = get_mean_number_of_source_over_4pi(S, dNdSdOmega, plot_fname="figs/mean_number_source")

Ntot_sources = np.sum(mean_numbers_per_patch)
print("Total number of radio sources", Ntot_sources)


# this get the monopole as a function of a flux cut on the sky, the flux cut can be seen as a masking parameter
# basically how much do you decrease the monopole if you mask all sources above S_max
monopole_fSmax = get_cumulative_monopole(S, dNdSdOmega, "figs/integrand.png")

plt.figure(figsize=(8,6))
plt.loglog()
plt.plot(S, monopole_fSmax)
plt.xlabel(r"$S_{\rm max}$ (Flux [Jy])", fontsize=fontsize)
plt.ylabel(r"$I_{\rm 148  \ GHz} [Jy . sr^{-1}]$", fontsize=fontsize)
plt.tick_params(labelsize=labelsize)
plt.savefig("figs/cumulative_monopole.png", bbox_inches="tight")
plt.clf()
plt.close()


monopole = monopole_fSmax[-1] # the actual monopole assuming no masking

freq_GHz = np.linspace(1,3000,1000)
alpha = -0.5
mono_radio =   monopole *  (freq_GHz / ref_freq_radio_GHz) ** alpha  # we assume the scaling of monopole is a power law with alpha = -0.5


# other distorsion computed using luca pagano code: https://github.com/paganol/BISOU-sky

nu, mudist = np.loadtxt("distorsion/mu.txt", unpack=True)
nu, ydist = np.loadtxt("distorsion/y.txt", unpack=True)
nu, relcorr = np.loadtxt("distorsion/relcorr.txt", unpack=True)

   
   
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

   
plt.figure(figsize=(8,6))
plt.loglog()
plt.plot(nu,-ydist,'--',lw=1,color=colors[0])
plt.plot(nu,ydist,'-',lw=1,color=colors[0],label="y distortions")
plt.plot(nu,-mudist,'--',lw=1,color=colors[1])
plt.plot(nu,mudist,'-',lw=1,color=colors[1],label=r"$\mu$ distortions")
plt.plot(nu,-relcorr,'--',lw=1,color=colors[2])
plt.plot(nu,relcorr,'-',lw=1,color=colors[2],label="Relativistic corrections")
plt.plot(freq_GHz, mono_radio,color=colors[3],label="radio sources")
plt.xlabel(r"$\nu$ [GHz]", fontsize=fontsize)
plt.ylabel(r"$I_{\nu} [Jy . sr^{-1}]$", fontsize=fontsize)
plt.tick_params(labelsize=labelsize)
plt.ylim(10**-1,10**5)
plt.legend(fontsize=labelsize)
plt.savefig("figs/monopole_vs_freq.png", bbox_inches="tight")
plt.clf()
plt.close()

