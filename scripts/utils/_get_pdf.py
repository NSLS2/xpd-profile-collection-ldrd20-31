from pdfstream.transformation.cli import transform
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pdfstream.io import load_array
from pdfstream.transformation.io import load_pdfconfig, write_pdfgetter
from pdfstream.transformation.main import get_pdf
from pathlib import Path
from diffpy.pdfgetx import PDFConfig
from diffpy.pdfgetx.pdfgetter import PDFConfigError
import typing

def globfilename(path, search_item='*.tiff'):
    filenames = glob.glob(path+search_item)
    inputnames=[filename.replace('\\','/') for filename in filenames]
    return(inputnames)

def transform_bkg(
    cfg_file,
    data_file: str,
    output_dir: str = ".",
    plot_setting: typing.Union[str, dict] = None,
    test: bool = False,
    gr_fn: str = '/home/xf28id2/Documents/test.gr',
) -> typing.Dict[str, str]:
    """Transform the data."""
    if isinstance(cfg_file,str):
        pdfconfig = load_pdfconfig(cfg_file)
    else:
        pdfconfig = cfg_file
    
    if type(data_file) is str:
        chi = load_array(data_file)
    elif type(data_file) is np.ndarray:
        chi = data_file
    try:
        pdfgetter = get_pdf(pdfconfig, chi, plot_setting=plot_setting)
    except PDFConfigError:
        pdfgetter = get_pdf(pdfconfig, chi, plot_setting='OFF')
    # filename = Path(data_file).stem
    # dct = write_pdfgetter(output_dir, filename, pdfgetter)
    dct = write_pdfgetter(output_dir, gr_fn, pdfgetter)
    if not test:
        plt.show()
    return dct


def count_header(filename):
    # count header 
    with open (filename,'r') as myfile:
        data = myfile.read().splitlines()
    count = 0
    for i, lines in enumerate(data):
        if lines == '#### start data':
            count =i
    return count+2

def find_qmax(filename, ave_cutoff=8e-03, window_length=21):
    
    header_count=count_header(filename)
    q, f = np.loadtxt(filename, skiprows= header_count, unpack=True)
    qmax = q[-1]
    fhat = savgol_filter(f, window_length, 3)
    fdiff=f-fhat
    zero_cross = np.where(np.diff(np.signbit(fhat)))[0]
    z=zero_cross[::-1]
    for i, i_zero in enumerate(z):
        ave=np.average(abs(fdiff[i_zero:]))

        if ave > ave_cutoff:
            print(i_zero, q[i_zero], ave)
        else:
            qmax = np.round(q[z[i-1]],1)
            print(i_zero, q[i_zero], ave)
            print(f'qmax ={qmax}')
            break    
    plt.figure(figsize=(16,8))            
    plt.plot(q,f)
    plt.plot(q, fhat, 'r')

    plt.plot(q, (f-fhat))   
    print(qmax)
    return qmax


if __name__ == "__main__":

    '''
    homepath='/home/xf28id2/Documents/ChengHung/pdfstream_test/'
    # testpath = './chi_files/'
    output_dir=homepath
    testfiles = globfilename(homepath, "*.chi")
    testfile =testfiles[0]
    bkg_file = testfiles[1]

    pdfconfig = PDFConfig()
    pdfconfig.wavelength = 0.18447
    pdfconfig.composition = 'CsPbBr3'
    #pdfconfig.readConfig(cfg_file)
    # config_file= globfilename('./','*.cfg')[0]
    # pdfconfig.readConfig(config_file)
    pdfconfig.backgroundfiles = bkg_file
    pdfconfig.bgscale=0.9
    print(pdfconfig)

    # transform('pdfconfig.cfg', testfile, output_dir=output_dir, plot_setting={'marker':'.','color':'green'} )
    # transform_bkg('pdfconfig.cfg', testfile, output_dir=output_dir, plot_setting={'marker':'.','color':'green'} )
    # transform_bkg(pdfconfig, testfile, output_dir=output_dir, plot_setting={'marker':'.','color':'green'} )

    # pdfconfig = PDFConfig()
    # config_file= globfilename('./','*.cfg')[0]
    # pdfconfig.readConfig(config_file)
    pdfconfig.qmaxinst = 24.0
    pdfconfig.qmin=1.0
    pdfconfig.qmax=20
    pdfconfig.rpoly=0.9
    pdfconfig.rmin=0.1
    pdfconfig.rmax=15
    # pdfconfig.rpoly=0.9
    pdfconfig.bgscale=1
    pdfconfig.outputtypes=['sq','fq','gr']
    print(pdfconfig)
    
    transform_bkg(pdfconfig, testfile, output_dir=output_dir, plot_setting={'marker':'.','color':'green'} )

    fqfile = globfilename(output_dir, '*/*.fq')[0]
    print(fqfile)

    qmax=find_qmax(fqfile,  ave_cutoff=8e-03, window_length=101)

    pdfconfig.qmax=qmax
    print(pdfconfig)
    transform_bkg(pdfconfig, testfile, output_dir=output_dir, plot_setting={'marker':'.','color':'green'} )
    '''



