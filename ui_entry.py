import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from typing import Callable
from tdcca.tdcca_parallel import TDCCAP
from tdcca.tdcca_cascade import TDCCAC
from tdpls.tdpls_parallel import TDPLSP
from tdpls.tdpls_cascade import TDPLSC
from helpers.correlation_helper import Correlation
from helpers.image_helper import ImageHelper
from abstract.classifier import ClassifierAlgorithm
from tkinter import ttk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
                
        self.training_folder = "dataset_old"
        self.testing_folder = "dataset_old"
    
    def create_widgets(self):
        self.tree = ttk.Treeview(self)
        self.tree["columns"] = ("X_test", "Y_train", "Y_test", "X_train")
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("X_test", anchor=tk.W, stretch=tk.YES)
        self.tree.column("Y_train", anchor=tk.W, stretch=tk.YES)
        self.tree.column("Y_test", anchor=tk.W, stretch=tk.YES)
        self.tree.column("X_train", anchor=tk.W, stretch=tk.YES)

        self.tree.heading("#0", text="", anchor=tk.W)
        self.tree.heading("X_test", text="X_test", anchor=tk.W)
        self.tree.heading("Y_train", text="Y_train", anchor=tk.W)
        self.tree.heading("Y_test", text="Y_test", anchor=tk.W)
        self.tree.heading("X_train", text="X_train", anchor=tk.W)

        # Add a scrollbar
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)

        self.tree.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side='right', fill='y')
        self.tree.pack(side="top", fill="both", expand=True)

        self.train_button = tk.Button(self)
        self.train_button["text"] = "Add Training Folder"
        self.train_button["command"] = self.load_training_folder
        self.train_button.pack(side="top")

        self.test_button = tk.Button(self)
        self.test_button["text"] = "Add Testing Folder"
        self.test_button["command"] = self.load_testing_folder
        self.test_button.pack(side="top")

        self.algorithm_var = tk.StringVar(self)
        self.algorithm_var.set("TDCCA(Parallel)")  # default value
        self.algorithm_option = tk.OptionMenu(self, self.algorithm_var, "TDCCA(Parallel)", "TDCCA(Cascade)", "TDPLS(Parallel)", "TDPLS(Cascade)")
        self.algorithm_option.pack(side="top")

        self.start_button = tk.Button(self)
        self.start_button["text"] = "Start Algorithm"
        self.start_button["command"] = self.start_algorithm
        self.start_button.pack(side="top")

        self.result_button = tk.Button(self)
        self.result_button["text"] = "Show Results"
        self.result_button["command"] = self.show_results
        self.result_button.pack(side="top")

        self.show_folders_button = tk.Button(self)
        self.show_folders_button["text"] = "Show Selected Folders"
        self.show_folders_button["command"] = self.show_selected_folders
        self.show_folders_button.pack(side="top")
        self.rrpp_var = tk.BooleanVar()
        self.rrpp_checkbutton = tk.Checkbutton(self, text="Use RRPP", variable=self.rrpp_var, command=self.toggle_d_entry)
        self.rrpp_checkbutton.pack(side="top")

        self.d_entry = tk.Entry(self)
        self.d_entry.insert(0, "10")  # default value
        self.d_entry.pack(side="top")
        self.d_entry.configure(state='disabled')  # disable by default

        self.quit = tk.Button(self, text="QUIT", fg="red",
                                command=self.master.destroy)
        self.quit.pack(side="bottom")
        
    def toggle_d_entry(self):
        if self.rrpp_var.get():
            self.d_entry.configure(state='normal')
        else:
            self.d_entry.configure(state='disabled')

    def load_training_folder(self):
        self.training_folder = filedialog.askdirectory()
        print(f"Training folder: {self.training_folder}")
        self.master.focus_set()

    def load_testing_folder(self):
        self.testing_folder = filedialog.askdirectory()
        print(f"Testing folder: {self.testing_folder}")
        self.master.focus_set()

    def show_selected_folders(self):
        training_folder = getattr(self, 'training_folder', 'Not selected')
        testing_folder = getattr(self, 'testing_folder', 'Not selected')
        messagebox.showinfo("Selected Folders", f"Training folder: {training_folder}\nTesting folder: {testing_folder}")
        self.master.focus_set()

    def start_algorithm(self):
        # Add your algorithm here
        algorithm = self.algorithm_var.get()
        d_value = 10
        try:
            d_value = int(self.d_entry.get())
        except Exception as e: d = 10
        messagebox.showinfo("Information",f"Algorithm: {algorithm}\nD value: {d_value}\nAlgorithm'll start after you close this window")
        self.master.focus_set()
        
        X_train, Y_train, train_links_x, train_links_y = self._get_data(num_test=2, dataset=self.training_folder)
        X_test, Y_test, test_links_x, test_links_y = self._get_data(num_test=4, dataset=self.testing_folder)
        method: ClassifierAlgorithm = None
        x_test_to_y_train, accuracy_x, correct_predictions_x = None, None, None
        y_test_to_x_train, accuracy_y, correct_predictions_y = None, None, None
        if algorithm == "TDCCA(Parallel)":
            method = self._fit_model(X_train, Y_train, TDCCAP)
        elif algorithm == "TDCCA(Cascade)":
            method = self._fit_model(X_train, Y_train, TDCCAC)
        elif algorithm == "TDPLS(Parallel)":
            method = self._fit_model(X_train, Y_train, TDPLSP)
        elif algorithm == "TDPLS(Cascade)":
            method = self._fit_model(X_train, Y_train, TDPLSC)
        else: messagebox.showerror("Error", "Unknown algorithm selected")
            
        x_test_to_y_train, accuracy_x, correct_predictions_x = self._predict_and_print(method, X_test, train_links_y, test_links_x, test_links_y, is_x=True)
        y_test_to_x_train, accuracy_y, correct_predictions_y = self._predict_and_print(method, Y_test, train_links_x, test_links_y, test_links_y, is_x=False)
        self.tree.delete(*self.tree.get_children())
        for x, y in zip(x_test_to_y_train, y_test_to_x_train):
            self.tree.insert("", "end", values=(x[0].removeprefix(self.training_folder), x[1].removeprefix(self.training_folder), 
                                                y[0].removeprefix(self.training_folder), y[1].removeprefix(self.testing_folder)))
        self.accuracy_x = accuracy_x
        self.accuracy_y = accuracy_y
        self.correct_predictions_x = correct_predictions_x
        self.correct_predictions_y = correct_predictions_y

    def show_results(self):
        try:
            accuracy = (self.accuracy_x + self.accuracy_y) / 2
            messagebox.showinfo("Information",f"Accuracy: {accuracy: .5f}\nCorrect_predictions(X_test): {self.correct_predictions_x}\nCorrect_predictions(Y_test): {self.correct_predictions_y}")
        except Exception as e:
            messagebox.showerror("Information", "Algorithm has not been run yet.")
        self.master.focus_set()
        
    def _get_data(self,num_test, dataset):
        links_x, links_y = ImageHelper.get_links(num_test=num_test, dataset=dataset)
        X = ImageHelper.get_pictures(links_x)
        Y = ImageHelper.get_pictures(links_y)
        return X, Y, links_x, links_y

    def _fit_model(self, X, Y, classifier: Callable):
        distantion = Correlation.distantion
        
        d = 10
        with_RRPP = False
        if self.rrpp_var.get():
            with_RRPP = True
            try:
                d = int(self.d_entry.get())
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Using default value for d: 10")
                d = 10
        
        method: ClassifierAlgorithm = classifier(dimension=d, distance_function=distantion, is_max=False)
        method.fit(X, Y, with_RRPP=with_RRPP)
        return method
    
    def _predict_and_print(self, algo:ClassifierAlgorithm, X_test, train_links_x: list[str], test_links_x, test_links_y, is_x=True):
        index_pair = algo.predict(X_test, is_x=is_x)
        correct_predictions = [index for index in range(len(index_pair)) if index == index_pair[index][0]]
        x_test_to_y_train = []
        for index in correct_predictions:
            print(train_links_x[index], test_links_x[index])
            x_test_to_y_train.append((train_links_x[index] , test_links_x[index]))
        
        total_predictions = len(index_pair)
        accuracy = round(len(correct_predictions) / total_predictions, 5)
        
        print(f"Number of correct predictions: {len(correct_predictions)}")
        print(f"Total predictions: {total_predictions}")
        print(f"Accuracy: {accuracy}")
        return x_test_to_y_train, accuracy, len(correct_predictions)
 


root = tk.Tk()
app = Application(master=root)
app.mainloop()