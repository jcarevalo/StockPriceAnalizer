import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib import style
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from DataManipulation import Grabber
from DataManipulation import StockMetricsCalculator
from DataManipulation import StockVisualizer
import datetime
from time import sleep


class GUI (tk.Tk):
    _LARGE_FONT = ("Verdana", 12)
    _NORM_FONT = ("Verdana", 10)
    _SMALL_FONT = ("Verdana", 8)

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        # tk.Tk.iconbitmap(self, default="myicon.ico")

        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in [StartPage, MainPage]:
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        menu_bar = tk.Menu(container)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Save settings", command=lambda: self.popup_msg("Not supported"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        indicator_choice = tk.Menu(menu_bar, tearoff=1)
        indicator_choice2 = tk.Menu(menu_bar, tearoff=1)
        indicators = ["ROC", "MACD", "RSI", "ADX"]
        for indicator in indicators:
            indicator_choice.add_command(label=indicator,
                                         command=lambda param=indicator: self.frames[MainPage].put_indicator(param, 'top'))
        for indicator in indicators:
            indicator_choice2.add_command(label=indicator,
                                          command=lambda param=indicator: self.frames[MainPage].put_indicator(param, 'bottom'))

        menu_bar.add_cascade(label="Top Indicators", menu=indicator_choice)
        menu_bar.add_cascade(label="Bottom Indicators", menu=indicator_choice2)

        ticker_choice = tk.Menu(menu_bar, tearoff=0)
        ticker_choice.add_command(label="Enter Ticker",
                                        command=lambda: self.get_stock())

        menu_bar.add_cascade(label="Stock", menu=ticker_choice)

        tk.Tk.config(self, menu=menu_bar)

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def get_stock(self):
        input_box = tk.Tk()
        input_box.wm_title('Which Stock')
        label = ttk.Label(input_box, text='Enter Ticker', font=self._NORM_FONT)
        label.pack(side="top", fill="x", pady=10)
        input_text = ttk.Entry(input_box)
        input_text.pack()
        input_text.delete(0)
        input_text.insert(0, "GOOG")

        def callback():
            ticker = input_text.get()
            canvas = self.frames[MainPage].canvas
            toolbar = self.frames[MainPage].toolbar

            if canvas is not None:
                self.frames[MainPage].visualizer.ax1.clear()# get_graph().clear()
                canvas.get_tk_widget().destroy()
                toolbar.destroy()

            self.frames[MainPage].draw_graph(ticker)
            input_box.destroy()

        b = ttk.Button(input_box, text="get", width=10, command=callback)
        b.pack()

        input_box.mainloop()

    def popup_msg(self, msg):
        popup = tk.Tk()
        popup.wm_title("!")
        label = ttk.Label(popup, text=msg, font=self._NORM_FONT)
        label.pack(side="top", fill="x", pady=10)
        ok_button = ttk.Button(popup, text="Okay", command=popup.destroy)
        ok_button.pack()
        popup.mainloop()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="""This program is meant for information and education exclusively. 
        I you want to use it to trade, do so at your own risk. I am not responsible for any decisions you make based on any information provided.
        The software is given as is and there is no promise of warranty.""", font=GUI._LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Agree", command=lambda: controller.show_frame(MainPage))
        button1.pack()

        button2 = ttk.Button(self, text="Disagree", command=quit)
        button2.pack()


class MainPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        button1 = ttk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.pack()
        self.grabber = None
        self.visualizer = None
        self.canvas = None
        self.toolbar = None
        self.top_ind = None
        self.bot_ind = None

    def put_indicator(self, params, where):

        if self.visualizer is not None:
            print(params)
            indicator_dic = {"ROC": self.visualizer.make_roc_plot, "MACD": self.visualizer.make_macd_plot,
                             "RSI": self.visualizer.make_rsi_plot, "ADX": self.visualizer.make_adx_plot}
            if where == 'top':
                self.visualizer.ax3.clear()  # get_graph().clear()
                indicator_dic[params](ax=self.visualizer.ax3)
                self.top_ind = indicator_dic[params]
            elif where == 'bottom':
                self.visualizer.ax2.clear()  # get_graph().clear()
                indicator_dic[params](ax=self.visualizer.ax2)
                self.bot_ind = indicator_dic[params]

            self.visualizer.make_axes()
            self.canvas.draw()

    def draw_graph(self, stock):
        self.grabber = Grabber([2015, 1, 1], [2018, 3, 31], stock)
        if self.visualizer is None:
            self.top_ind = lambda ax=None: self.visualizer.make_macd_plot()
            self.bot_ind = lambda ax=None: self.visualizer.make_roc_plot()
        self.visualizer = StockVisualizer(self.grabber, 60)
        self.visualizer.make_candle_stick(ma_list=[50])
        self.visualizer.make_volume_plot(overlay=True)
        self.top_ind(ax=self.visualizer.ax3)
        self.bot_ind(ax=self.visualizer.ax2)
        self.fig = self.visualizer.get_graph()
        self.canvas = FigureCanvasTkAgg(self.visualizer.get_graph(), self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = matplotlib.backends.backend_tkagg.NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()

        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def animate(frames):
            end = datetime.datetime.today() + datetime.timedelta(1)
            start = datetime.datetime.today() - datetime.timedelta(600)
            self.visualizer.update_data(start, end)
            self.visualizer.ax1.clear()
            self.visualizer.ax2.clear()
            self.visualizer.ax3.clear()
            if self.visualizer is None:
                self.top_ind = lambda ax=None: self.visualizer.make_macd_plot()
                self.bot_ind = lambda ax=None: self.visualizer.make_roc_plot()
            self.visualizer.make_candle_stick(ma_list=[50], ax=self.visualizer.ax1)
            self.visualizer.make_volume_plot(overlay=True, ax=self.visualizer.ax_twin)
            self.top_ind(ax=self.visualizer.ax3)
            self.bot_ind(ax=self.visualizer.ax2)

            self.canvas.draw()

        self.animation = FuncAnimation(self.fig, animate, interval=30000)




app = GUI()
app.geometry("1280x600")
# ani = FuncAnimation(fig, animate, frames=1000)
app.mainloop()
# if __name__ == '__main__':
#     print(datetime.datetime.today().month)