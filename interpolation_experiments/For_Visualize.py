import matplotlib.pyplot as plt
import pickle
import os

def visual_interpol(start, end, interpol_arr):
	plt.rc('xtick', labelsize=15) 
	plt.rc('ytick', labelsize=15) 
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.tick_params(direction='in',width=4,length=8)
	# lin_style = 'densely dashdotdotted'
	lin_style = (0, (5,1))
	c_0 = 20
	c_1 = 40
	c_2 = len(interpol_arr)//2
	c_3 = len(interpol_arr)-40
	c_4 = len(interpol_arr)-20
	print((c_1,c_2,c_3))
	c_max = len(interpol_arr)
	plt.title("Interpolation" , fontsize = 20 )
	plt.xlabel('time', fontsize = 20)
	plt.ylabel('value', fontsize = 20)
	# plt.plot(start)
	plt.plot(interpol_arr[0])
	plt.plot(interpol_arr[5], color = (c_0/c_max,c_0/c_max ,c_0/c_max), linestyle = lin_style )
	plt.plot(interpol_arr[20], color = (c_1/c_max,c_1/c_max ,c_1/c_max) , linestyle = lin_style)
	plt.plot(interpol_arr[len(interpol_arr)//2], color =(c_2/c_max,c_2/c_max,c_2/c_max), linestyle = lin_style )
	plt.plot(interpol_arr[-20], color = (c_3/c_max,c_3/c_max,c_3/c_max), linestyle = lin_style )
	plt.plot(interpol_arr[-5], color = (c_4/c_max,c_4/c_max ,c_4/c_max), linestyle = lin_style )
	plt.plot(interpol_arr[-1])
	# plt.plot(end)
	plt.show()


def visual_reconstr(signal_for_reconst, reconstr_arr, frames, size_of_data):
    from matplotlib.animation import FuncAnimation
    fig = plt.figure(3)
    ax1 = fig.add_subplot(1, 1, 1)
    def animate(i):
        x = np.arange(0,size_of_data)
        y = reconstr_arr[i].reshape(size_of_data)
        ax1.clear()
        ax1.plot(signal_for_reconst.reshape(size_of_data))
        ax1.plot(x, y)
        plt.xlabel('time')
        plt.ylabel('signal')
        plt.title("Epoch =  "+ str(i+1))
        plt.legend(['signal', 'output from nn'], loc='upper left')
    anim = FuncAnimation(fig, animate,frames=frames, interval=100)
    anim.save('animation_reconstr.gif', writer='imagemagick', fps=60)
    plt.show()

with open('history of learning AE\\interpol_array0.pickle', 'rb') as z:
	interpol_data = pickle.load(z)

with open('history of learning AE\\interpol_array3.pickle', 'rb') as z:
	interpol_data2 = pickle.load(z)

with open('history of learning AE\\reconstr.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
	reconstr_signal , reconstr  = loaded_data


# visual_interpol(*interpol_data)
# visual_interpol(*interpol_data2)
# visual_reconstr(reconstr_signal, reconstr, np.shape(reconstr)[0], np.shape(reconstr_signal)[1])
