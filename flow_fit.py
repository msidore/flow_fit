#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, subprocess, os, math
import numpy as np
from argparse import ArgumentParser
#~ import matplotlib
#~ matplotlib.use('PS')
import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import linregress
# Ignore warnings - ofc you can't divide by 0
import warnings
warnings.filterwarnings("ignore")

"""

To use with the stopflow machine

Usage:
./flow_fit.py -f xx.csv -o outname -s -sg -b

"""

######################## Parsing stuff ########################

#~ directory = ""
bubble = False
savgol = False

parser = ArgumentParser(description=""" To analyze stopflow data """)

# Named arguments
group = parser.add_mutually_exclusive_group(required=True)

# If exp_type is True, then it's a simple exponent - else it's False
group.add_argument("-s", "--simple", required=False, dest="exp_type", action="store_true", help="Add this option to fit with the simple exponent a1*np.exp(-b1/t1) + baseline")
group.add_argument("-d", "--double", required=False, dest="exp_type", action="store_false", help="Add this option to fit with the double exponent a1*np.exp(-b1/t1) + a2*np.exp(-b2/t2) + baseline")

# The file and the output name
parser.add_argument("-f", "--file", type=str, required=True, dest="infile", help="The name of the input file")
parser.add_argument("-o", "--output", type=str, required=True, dest="outfile", help="The generic output name")

# Does your experiment have a bubble ? Will skip the first minima
parser.add_argument("-b", "--bubble", required=False, dest="bubble", action="store_true", help="Use this option if your experiment has a bubble. Will skip the first minima and do the analysis from the second")

# Do you want to smooth your data ? It's gonna use the Savitsky Golay method
parser.add_argument("-sg", "--savgol", required=False, dest="savgol", action="store_true", help="Use this option if you want to smooth your data using the Savitsky Golay method with a 2nd order derivative")

args = parser.parse_args()

######################## Directory stuff ########################

#~ if args.directory:
    #~ # Checks if the directory has a /
    #~ if args.directory[-1] == "/":
        #~ directory += args.directory
    #~ else:
        #~ directory += args.directory + "/"
    #~ subprocess.Popen("mkdir " + directory, shell=True, stderr=subprocess.PIPE).wait()

######################## Functions and miscellaneous ########################

infile = args.infile
outfile = args.outfile
exp_type = args.exp_type
bubble = args.bubble
savgol = args.savgol

# Use the right template for simple/double exponent
if exp_type == True:
    TEMPLATE = """
    %s \t %s \t %s \t %s \t %s \t %s \t %s
    """
else:
    TEMPLATE = """
    %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s
    """

def simple_exponent(t, a1, b1, baseline):
    t1 = t
    return a1*np.exp(-b1/t1) + baseline

def double_exponent(t, a1, a2, b1, b2, baseline):
    t1 = t
    t2 = t
    return a1*np.exp(-b1/t1) + a2*np.exp(-b2/t2) + baseline

######################## Main ########################

if __name__ == '__main__':

    # Open/create the file that will contain the parameters
    textfile = open(outfile, "w")
    if exp_type == True:
        textfile.write("name \t a1 \t a1-stdev \t Tau1 \t baseline \t baseline-stdev \t R")
    else:
        textfile.write("name \t a1 \t a1-stdev \t a2 \t a2-stdev \t Tau1 \t Tau2 \t baseline \t baseline-stdev \t R")
    # Open/create the file that will contain the residuals and raw/smoothed data
    textfile_data = open("%s_data" % outfile, "w")

    # Open the file that has all the data
    infile_txt = open(infile, "r").read().replace(",", ".")

    # Get the lines - and remove the empty lines - and make sure to remove the possibly residual \n
    fic = [x for x in infile_txt.split('\r') if x != ""]
    fic = [x.strip("\n") for x in fic]
    # If the file has \n instead of \r, it's a problem, so we'll check if the split has been good, if the number of lines we get is less than, say, 5
    if len(fic) <= 5:
        fic = [x for x in infile_txt.split('\n') if x != ""]
        # This next line will probably be useful, I dunno
        #~ fic = [x.strip("\r") for x in fic]

    # Now we should have a good splitted file - the first line should then contain the title
    title = fic[0].split(";")[0]
    # And we actually don't care about the first two lines now, they contain only the title and the column titles such as "wavelength" etc
    fic = fic[2:]
    # Split each line with the ";" separator
    fic2 = [x.split(";")[:-1] for x in fic]

    # This file contains multiple experiments
    # Rearrange the thing into these different experiments
    # First get the number of experiments
    tmp = int(len(fic2[0])/2)

    # And now rearrange things - create an empty list
    fic3 = []
    # Loop through the different experiments to have each occurence in the list fic3 being one experiment
    for i in range(0, tmp*2, 2):
        # Create an empty sublist that will have every line of the experiment
        sublist = []
        # Add the right experiment to the sublist
        for j in range(len(fic2)):
            sublist.append(fic2[j][i:i+2])
        # And append the sublist, which has all the lines from one experiment, to fic3
        fic3.append(sublist)

    # Now fic3 is a list of two columns files [[[aA1, aA2],[bA1, bA2],...[zA1, zA2]][[aB1, aB2],[bB1, bB2],...,[zB1, zB2]]]

    # Get a simple list of the experiment numbers
    outnum = [str(x+1) for x in list(range(tmp))]

    # Now we can treat each experiment separately !
    # Loop through all the experiments
    for ite in range(len(fic3)):

        # Get the data
        # t1 is the array corresponding to the times of the data points
        if len(fic3[ite][-1]) != 0:
            if fic3[ite][-1][0] == '':
                # This condition is for badly formatted, out of experiment files
                t1 = np.array([float(x[0]) for x in fic3[ite] if x[0]!=''])
                noisy1 = np.array([0.000001*float(x[1]) for x in fic3[ite] if x[0]!=''])
            elif len(fic3[ite][-1]) == 2:
                # This condition is actually for rightly formatted csv files
                t1 = np.array([float(x[0]) for x in fic3[ite] if x[0]!=''])
                noisy1 = np.array([0.000001*float(x[1]) for x in fic3[ite] if x[0]!=''])
            else:
                # And this is if there are even stranger things happening with the formatting
                print "Error with formatting, exiting ..."
                sys.exit()
        else:
            t1 = np.array([float(x[0]) for x in fic3[ite] if len(x)!=0])
            noisy1 = np.array([0.000001*float(x[1]) for x in fic3[ite] if len(x)!=0])
        # noisy1 is the data array - which is noisy, which is why its name is "noisy1" - *10^-6 because the values are so high

        # Split at the first minima
        for x in range(len(noisy1)):
            if noisy1[x] == min(noisy1):
                t = t1[x:]
                noisy = noisy1[x:]

        # Shift the x
        t = [x-t[0] for x in t]

        # Split at the second minima ?
        if bubble == True:
            # First discard the first five points (because a bubble is even more noisy)
            noisy3 = noisy[5:]
            t3 = t[5:]
            # Then seek the second minima
            for x in range(len(noisy3)):
                if noisy3[x] == min(noisy3):
                    t = t3[x:]
                    noisy = noisy3[x:]
            # Reshift the x
            t = [x-t[0] for x in t]

        # Savitsky-Golay filter ? 5 is the window length, 2 is the exponent here
        if savgol == True:
            data = savgol_filter(noisy, 5, 2)
        else:
            data = noisy

        # Now the curve fitting part - depends if it's with a simple or a double exponent
        if exp_type == True:
            # Get the parameters and the covariance matrix from the fit
            fitParams, fitCovariances = curve_fit(simple_exponent, t, data, maxfev=5000000, p0=[1, 1, np.mean(data)])
            # Get the standard deviations of the parameters from the covariance matrix
            perr = np.sqrt(np.diag(fitCovariances))

            # Get the R squared and a yarray from the function
            yarray = [simple_exponent(time, fitParams[0], fitParams[1], fitParams[2]) for time in t]
            r = linregress(data, yarray)[2]
            rsquared = str(r**2)

            # Write the parameters into the outfile
            textfile.write(TEMPLATE % (outnum[ite], fitParams[0], perr[0], fitParams[1], fitParams[2], perr[2], rsquared))
            # And the data - with or without smoothing
            textfile_data.write("Experiment %s\n" % outnum[ite])
            if savgol == True:
                # Write the header
                textfile_data.write("Residuals;Fit;Raw data;Smoothed data\n")
                for i in range(len(t)):
                    # Get the fitted value with the exponent function and the parameters
                    y_fit = simple_exponent(t[i], fitParams[0], fitParams[1], fitParams[2])
                    # Get the residual
                    y_residual = noisy[i] - y_fit
                    # Write out everything
                    textfile_data.write(str(y_residual) + ";" + str(y_fit) + ";" + str(noisy[i]) + ";" + str(data[i]) + "\n")
                textfile_data.write("\n")
            else:
                textfile_data.write("Residuals;Fit;Raw data\n")
                for i in range(len(t)):
                    y_fit = simple_exponent(t[i], fitParams[0], fitParams[1], fitParams[2])
                    y_residual = noisy[i] - y_fit
                    textfile_data.write(str(y_residual) + ";" + str(y_fit) + ";" + str(noisy[i]) + "\n")
                textfile_data.write("\n")
        else:
            # Get the parameters and the covariance matrix from the fit
            fitParams, fitCovariances = curve_fit(double_exponent, t, data, maxfev=5000000, p0=[1, 1, 1, 1, np.mean(data)])
            # Get the standard deviations of the parameters from the covariance matrix
            perr = np.sqrt(np.diag(fitCovariances))

            # Get the R squared and a yarray from the function
            yarray = [double_exponent(time, fitParams[0], fitParams[1], fitParams[2], fitParams[3], fitParams[4]) for time in t]
            r = linregress(data, yarray)[2]
            rsquared = str(r**2)

            # Write the parameters into the outfile
            textfile.write(TEMPLATE % (outnum[ite], fitParams[0], perr[0], fitParams[1], perr[1], fitParams[2], fitParams[3], fitParams[4], perr[4], rsquared))
            # And the data - with or without smoothing
            textfile_data.write("Experiment %s\n" % outnum[ite])
            if savgol == True:
                # Write the header
                textfile_data.write("Residuals;Fit;Raw data;Smoothed data\n")
                # Loop through all time points
                for i in range(len(t)):
                    # Get the fitted value with the exponent function and the parameters
                    y_fit = double_exponent(t[i], fitParams[0], fitParams[1], fitParams[2], fitParams[3], fitParams[4])
                    # Get the residual
                    y_residual = noisy[i] - y_fit
                    # Write out everything
                    textfile_data.write(str(y_residual) + ";" + str(y_fit) + ";" + str(noisy[i]) + ";" + str(data[i]) + "\n")
                textfile_data.write("\n")
            else:
                textfile_data.write("Residuals;Fit;Raw data\n")
                for i in range(len(t)):
                    y_fit = double_exponent(t[i], fitParams[0], fitParams[1], fitParams[2], fitParams[3], fitParams[4])
                    y_residual = noisy[i] - y_fit
                    textfile_data.write(str(y_residual) + ";" + str(y_fit) + ";" + str(noisy[i]) + "\n")
                textfile_data.write("\n")

        # Makes figures
        plt.figure(outnum[ite])
        # With a title
        plt.title(title + "_" + outnum[ite])
        # A label to the x
        plt.xlabel("Time(s)")
        # A label to the y
        plt.ylabel("Intensity, *10^6")
        # The fitted curve
        plt.plot(t, yarray)
        # The original data
        plt.plot(t, noisy)
        plt.savefig(outnum[ite] + '.ps')
        plt.close(outnum[ite])

    # Cat and save the .ps into a .pdf
    ps_list = " ".join([outnum[ite] + '.ps' for ite in range(len(outnum))])
    bash_command = "cat " + ps_list + "> " + outfile + 'stopflow.ps'
    os.system(bash_command)
    bash_command = "epstopdf " + "--exact " + outfile + "stopflow.ps"
    os.system(bash_command)
