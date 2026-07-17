import matplotlib.pyplot as plt
import numpy as np
from _data_export import _readable_time
import _data_analysis as da

plt.ion()
plt.rcParams["figure.raise_window"] = False

class open_figures():
    def __init__(self, figure_labels):
        for i in figure_labels:
            plt.figure(num=i, figsize=(8,6))


class plot_uvvis(open_figures):
    
    def __init__(self, qepro_dic, metadata_dic, 
                 figure_labels = [
                    #  'primary_absorbance', 'primary_fluorescence', 
                     'bundle_absorbance', 'bundle_fluorescence',
                     'peak fitting', 'Spectra Evolution', 
                    #  'CsPbX3', 
                    #  'I(Q)', 'g(r)', 
                     ]
                 ):
        self.fig = figure_labels
        self.uid = metadata_dic['uid']
        self.stream_name = metadata_dic['stream_name']
        self.qepro_dic = qepro_dic
        self.metadata_dic = metadata_dic
        self.wavelength = []
        self.output = []
        self.fontsize = 9
        self.legend_properties = {'weight':'regular', 'size':8}
        # self.num = None
        self.date, self.time = _readable_time(metadata_dic['time'])
        super().__init__(figure_labels)


    def plot_data(self, label=None, title=None, clear_fig=False):
        self.wavelength = self.qepro_dic['QEPro_x_axis']
        self.output = self.qepro_dic['QEPro_output']
        
        global ax, y_label
        if self.stream_name == 'absorbance':
            y_label = 'Absorbance'
            f = plt.figure('bundle_absorbance')
            plt.clf()
            ax = f.gca()
        
        elif self.stream_name == 'fluorescence':
            y_label = 'Fluorescence'
            f = plt.figure('bundle_fluorescence')
            plt.clf()
            ax = f.gca()
        
        # elif (self.stream_name == 'take_a_uvvis') or (self.stream_name == 'primary'):
        #     if self.qepro_dic['QEPro_spectrum_type'] == 3:
        #         y_label = 'Absorbance'
        #         try: f = plt.figure(self.fig[0])
        #         except (IndexError): f = plt.figure(self.fig[-1])
        #         if clear_fig:
        #             plt.clf()
        #         ax = f.gca()
        #     else:
        #         y_label = 'Fluorescence'
        #         try: f = plt.figure(self.fig[1])
        #         except (IndexError): f = plt.figure(self.fig[-1])
        #         if clear_fig:
        #             plt.clf()
        #         ax = f.gca()

        for i in range(self.wavelength.shape[0]):
            if label == None:
                label = f'{self.time}_{i:03d}'
            ax.plot(self.wavelength[i], self.output[i], label=label)
            label = None
        # ax.set_facecolor((0.95, 0.95, 0.95))
        ax.set_xlabel('Wavelength (nm)', fontdict={'size': self.fontsize})
        ax.set_ylabel(y_label, fontdict={'size': self.fontsize})
        if title == None:
            title = f'{self.date}-{self.time}_{self.uid[0:8]}_{self.stream_name}'
        ax.set_title(title, fontdict={'size': self.fontsize})
        ax.tick_params(axis='both', labelsize=self.fontsize)
        ax.legend(prop=self.legend_properties)
        f.canvas.manager.show()
        f.canvas.draw_idle()
        f.canvas.flush_events()
        
        


    def plot_peak_fit(self, x, y, fit_function, popt, peak=None, fill_between=False):
        y_label = 'Fluorescence'
        
        f = plt.figure('peak fitting')
        
        plt.clf()
        ax = f.gca()
        
        fitted_y = fit_function(x, *popt)
        r2_idx1, _ = da.find_nearest(x, popt[1] - 3*popt[2])
        r2_idx2, _ = da.find_nearest(x, popt[1] + 3*popt[2])
        r_2 = da.r_square(x[r2_idx1:r2_idx2], y[r2_idx1:r2_idx2], fitted_y[r2_idx1:r2_idx2], y_low_limit=0)  
        # r_2 = da.r_square(x, y, fitted_y)
        r2 = f'R\u00b2={r_2:.2f}'
        ax.plot(x,y,'b+:',label='data')
        ax.plot(x[r2_idx1:r2_idx2], fitted_y[r2_idx1:r2_idx2],'r--',label='Total fit\n'+r2)
        
        try:
            for i in range(len(peak)):
                ax.plot(x[peak[i]], y[peak[i]], '*', markersize=12)
        except (TypeError, IndexError):
            pass
        
        if fill_between:
            if 'gauss' in fit_function.__name__:
                f1 = da._1gauss
                for i in range(int(len(popt)/3)):
                    pars_i = popt[0+3*i:3+3*i]
                    peak_i = f1(x, *pars_i)
                    ax.plot(x[r2_idx1:r2_idx2], peak_i[r2_idx1:r2_idx2], label=f'peak {i+1}')
                    ax.fill_between(x, peak_i.min(), peak_i, alpha=0.3)
            else:
                print(f'\n** Plot fill_between for {fit_function.__name__} is not supported. ** \n')

        # ax.set_facecolor((0.95, 0.95, 0.95))
        ax.set_xlabel('Wavelength (nm)', fontdict={'size': self.fontsize})
        ax.set_ylabel(y_label, fontdict={'size': self.fontsize})
        ax.set_title(f'{self.date}-{self.time}_{self.uid[0:8]}_{self.stream_name}_{fit_function.__name__}', fontdict={'size': self.fontsize})
        ax.tick_params(axis='both', labelsize=self.fontsize)
        ax.legend(prop=self.legend_properties)
        f.canvas.manager.show()
        f.canvas.draw_idle()
        f.canvas.flush_events()
        
        


    def plot_average_good(self, x, y, color=None, label=None, clf_limit=10):        
        
        # import palettable.colorbrewer.diverging as pld
        # palette = pld.RdYlGn_4_r
        # cmap = palette.mpl_colormap

        y_label = 'Fluorescence'
        
        f = plt.figure('Spectra Evolution')
        
        ax = f.gca()      
        if len(list(ax.lines)) > clf_limit:
            plt.clf()

        ax = f.gca()

        if label == None:
            label = self.time + '_' + self.uid[:8]

        if color == None:
            ax.plot(x, y, label=label)
        else:
            ax.plot(x, y, label=label, color=color)

        # ax.set_facecolor((0.95, 0.95, 0.95))
        ax.set_xlabel('Wavelength (nm)', fontdict={'size': self.fontsize})
        ax.set_ylabel(y_label, fontdict={'size': self.fontsize})
        # ax.set_title(f'{self.date}-{self.time}_{self.uid[0:8]}_{self.stream_name}_{fit_function.__name__}')
        ax.legend(prop=self.legend_properties)
        ax.tick_params(axis='both', labelsize=self.fontsize)
        f.canvas.manager.show()
        f.canvas.draw_idle()
        f.canvas.flush_events()





    def plot_CsPbX3(self, x, y, wavelength, label=None, clf_limit=10):        
        
        # import palettable.colorbrewer.diverging as pld
        # palette = pld.RdYlGn_4_r
        # cmap = palette.mpl_colormap

        y_label = 'Fluorescence'
        
        try:
            f = plt.figure(self.fig[6])
        except (IndexError):
            f = plt.figure(self.fig[-1])
        
        ax = f.gca()      
        if len(list(ax.lines)) > clf_limit:
            plt.clf()

        ax = f.gca()

        if label == None:
            label = self.time + '_' + self.uid[:8]

        # cmap = palette.mpl_colormap
        # cmap = plt.get_cmap('jet')
        # w1 = wavelength_range[0]  ## in nm
        # w2 = wavelength_range[1]  ## in nm
        # w_steps = abs(int(2*(w2-w1)))
        # w_array = np.linspace(w1, w2, w_steps)
        # color_array = np.linspace(0, 1, w_steps)
        # idx, _ = da.find_nearest(w_array, wavelength)
        # ax.plot(x, y, label=label, color=cmap(color_array[idx]))

        ax.plot(x, y, label=label, color=color_idx_map_halides(wavelength))

        # ax.set_facecolor((0.95, 0.95, 0.95))
        ax.set_xlabel('Wavelength (nm)', fontdict={'size': self.fontsize})
        ax.set_ylabel(y_label, fontdict={'size': self.fontsize})
        # ax.set_title(f'{self.date}-{self.time}_{self.uid[0:8]}_{self.stream_name}_{fit_function.__name__}')
        ax.tick_params(axis='both', labelsize=self.fontsize)
        ax.legend(prop=self.legend_properties)
        f.canvas.manager.show()
        f.canvas.draw_idle()
        f.canvas.flush_events()


    
    def plot_offfset(self, x, fit_function, popt):        
        
        # import palettable.colorbrewer.diverging as pld
        # palette = pld.RdYlGn_4_r
        # cmap = palette.mpl_colormap
        
        f = plt.figure('bundle_absorbance')
        
        ax = f.gca()

        ax.plot(x, fit_function(x, *popt), label=f'check baseline: {fit_function.__name__}\ny={popt[0]:.4f}x+{popt[1]:.4f}')

        # # ax.set_facecolor((0.95, 0.95, 0.95))
        # ax.set_xlabel('Wavelength (nm)', fontdict={'size': 14})
        # ax.set_ylabel(y_label, fontdict={'size': 14})
        # # ax.set_title(f'{self.date}-{self.time}_{self.uid[0:8]}_{self.stream_name}_{fit_function.__name__}')
        ax.legend(prop=self.legend_properties)
        f.canvas.manager.show()
        f.canvas.draw_idle()
        f.canvas.flush_events()

        
        
    def plot_iq_to_gr(self, iq_df, gr_df, gr_fit=None, label=None):
        try: f = plt.figure(self.fig[7])
        except (IndexError): f = plt.figure(self.fig[-1])
        plt.clf()
        ax = f.gca()
        if label == None:
            label = f'{self.time}_{self.uid[0:8]}'
        ax.plot(iq_df[0], iq_df[1], label=label)
        ax.set_xlabel('Q(A-1)', fontdict={'size': self.fontsize})
        ax.set_ylabel('I(Q)', fontdict={'size': self.fontsize})
        ax.legend(prop=self.legend_properties)
        
        try: f = plt.figure(self.fig[8])
        except (IndexError): f = plt.figure(self.fig[-1])
        plt.clf()
        ax = f.gca()
        if label == None:
            label = f'{self.time}_{self.uid[0:8]}'
        ax.plot(gr_df[0], gr_df[1], 'o', label=label, 
                markersize=4, markerfacecolor='none', markeredgewidth=0.4)
        if type(gr_fit) is np.ndarray:
            ax.plot(gr_fit[0], gr_fit[1], label='PDF Fit')
        ax.set_xlabel('r(A)', fontdict={'size': self.fontsize})
        ax.set_ylabel('g(r)', fontdict={'size': self.fontsize})
        ax.tick_params(axis='both', labelsize=self.fontsize)
        ax.legend(prop=self.legend_properties)
        
        # ax.set_ylabel(y_label, fontdict={'size': 14})
        # if title == None:
        #     title = f'{self.date}-{self.time}_{self.uid[0:8]}_{self.stream_name}'
        # ax.set_title(title)
        f.canvas.manager.show()
        f.canvas.draw_idle()
        f.canvas.flush_events()
        
        
        

        

class open_subfigures():
    def __init__(self, rows, columns, figsize, ax_titles):
        f1, ax1 = plt.subplots(rows, columns, figsize=figsize, constrained_layout=True)
        ax1 = ax1.flatten()
        for i in range(len(ax_titles)):
            ax1[i].set_title(ax_titles[i], fontsize=10)
        
class multipeak_fitting(open_subfigures):
    def __init__(self, rows=2, columns=2, figsize = (8, 6), ax_titles=['_1gauss', '_2gauss', '_3gauss', 'r_2']):
        super().__init__(rows, columns, figsize, ax_titles)
        self.fig = plt.gcf()
        self.ax = self.fig.get_axes()
        
    def plot_fitting(self, x, y, popt_list, single_f, fill_between=True, num_var=3):
        for i in range(len(popt_list)):
            
            fitted_y = np.zeros([x.shape[0]])
            for j in range(i+1):
                fitted_y += single_f(x, *popt_list[i][0+num_var*j:num_var+num_var*j])
            
            self.ax[i].plot(x ,y, 'b+:',label='data')
            r_2 = da.r_square(x, y, fitted_y, y_low_limit=500)
            r2 = f'R\u00b2={r_2:.2f}'
            self.ax[i].plot(x,fitted_y,'r--',label='Total fit\n'+r2)
            
            if fill_between:
                f1 = single_f
                for k in range(int(len(popt_list[i])/3)):
                    pars_k = popt_list[i][0+3*k:3+3*k]
                    peak_k = f1(x, *pars_k)
                    self.ax[i].plot(x, peak_k, label=f'peak {k+1}')
                    self.ax[i].fill_between(x, peak_k.min(), peak_k, alpha=0.3)
            
            self.ax[i].legend()
            





def color_idx_map_halides(peak_wavelength, halide_w_range=[400, 520, 660]):
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        # (0.25098039215686274, 0.0, 0.29411764705882354), 
        (0.4627450980392157, 0.16470588235294117, 0.5137254901960784),
        (0.3686274509803922, 0.30980392156862746, 0.6352941176470588), 
        (0.19607843137254902, 0.5333333333333333, 0.7411764705882353), 
        (0.4, 0.7607843137254902, 0.6470588235294118),
     # (0.6705882352941176, 0.8666666666666667, 0.6431372549019608),
     # (0.6509803921568628, 0.8509803921568627, 0.41568627450980394),
     # (0.10196078431372549, 0.5882352941176471, 0.2549019607843137), 
             ]
    BlGn = LinearSegmentedColormap.from_list('BlGn', colors, N=100)


    import palettable
    palette2 = palettable.colorbrewer.diverging.RdYlGn_4_r
    RdYlGn_4_r = palette2.mpl_colormap
    
    if  peak_wavelength >= halide_w_range[0] and peak_wavelength < halide_w_range[1]:
        wavelength_range=[halide_w_range[0], halide_w_range[1]]
        cmap = BlGn
    elif peak_wavelength >= halide_w_range[1] and peak_wavelength <= halide_w_range[2]:
        wavelength_range=[halide_w_range[1], halide_w_range[2]]
        cmap = RdYlGn_4_r
        
    else:
        raise ValueError(f'Peak at {peak_wavelength} nm is not in the range of {halide_w_range} nm.')
    
    w1 = wavelength_range[0]  ## in nm
    w2 = wavelength_range[1]  ## in nm
    w_steps = abs(int(2*(w2-w1)))
    w_array = np.linspace(w1, w2, w_steps)
    color_array = np.linspace(0, 1, w_steps)
    idx, _ = da.find_nearest(w_array, peak_wavelength)
    # ax.plot(x, y, label=label, color=cmap(color_array[idx]))
    
    color = cmap(color_array[idx])
    
    return color



class plot_callback(open_figures):
    
    def __init__(self, 
                 figure_labels = ['Test data']):
        
        self.fig = figure_labels
        # self.uid = metadata_dic['uid']
        # self.stream_name = metadata_dic['stream_name']
        # self.qepro_dic = qepro_dic
        # self.metadata_dic = metadata_dic
        # self.wavelength = []
        # self.output = []
        self.fontsize = 9
        self.legend_properties = {'weight':'regular', 'size':8}
        # self.num = None
        # self.date, self.time = _readable_time(metadata_dic['time'])
        super().__init__(figure_labels)


    def plot_data(self, x=None, y=None, label=None, title=None, clear_fig=False):

        try: 
            f = plt.figure(self.fig[0])
        except (IndexError): 
            f = plt.figure(self.fig[-1])
        
        plt.clf()
        ax = f.gca()
        ax.plot(x, y, label=label)

        # if title == None:
        #     title = f'{self.date}-{self.time}_{self.uid[0:8]}_{self.stream_name}'
        
        ax.set_title(title, fontdict={'size': self.fontsize})
        ax.tick_params(axis='both', labelsize=self.fontsize)
        ax.legend(prop=self.legend_properties)
        f.canvas.manager.show()
        f.canvas.draw_idle()
        f.canvas.flush_events()

