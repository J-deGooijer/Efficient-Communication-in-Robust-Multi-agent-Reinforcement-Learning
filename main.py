from train_agent import Train
import numpy as np
import pickle
import matplotlib.pyplot as plt

alpha = [0.1, 0.2, 0.5, 1, 2, 5, 10]

if __name__ == '__main__':

    # while True:
    #     try:
    # train_agents = Train('simple_tag')
    # for a in alpha:
    #     print("Training with alpha = ", a)
    #     train_agents.training(edi_mode='train', edi_load=False, alpha=a)

    # history = train_agents.testing(edi_mode='disabled', render=False)
    # mean = np.mean(history, axis=0)
    # std = np.std(history, axis=0)

    # alpha = [0.1, 0.2, 0.5, 1, 2, 5, 10]

    # for a in alpha:
    #     print("Testing with alpha = ", a)
    #     history = train_agents.testing(edi_mode='test', render=True, alpha=a)
    #     mean = np.vstack((mean, np.mean(history, axis=0)))
    #     std = np.vstack((std, np.std(history, axis=0)))
    #     print('Press enter')
    #     response = input()

    # with open('results.pickle', 'wb+') as f:
    #     pickle.dump([alpha, mean, std],f)

    

    with open('results.pickle', 'rb') as f:
        data = pickle.load(f)

    fig,ax = plt.subplots()
    # ax.fill_between(data[0], data[1][1:,0]+data[2][1:,0], data[1][1:,0]+data[2][1:,0], color="red", alpha=0.3)
    ax.plot(data[0], data[1][1:,0], color="red", marker="o")
    ax.set_xlabel("$\zeta_{\mathrm{th}}$", fontsize=14)
    ax.set_ylabel("Return adversaries", color="red", fontsize=14)
    ax.set_facecolor('#ecf0f1')

    ax2=ax.twinx()
    # ax2.fill_between(data[0], data[1][1:,2]+data[2][1:,2], data[1][1:,0]+data[2][1:,2], color="blue", alpha=0.3)
    ax2.plot(data[0], data[1][1:,2], color="blue", marker="o")
    ax2.set_ylabel("Communications", color="blue", fontsize=14)
    ax2.set_facecolor('#ecf0f1')
    plt.title("Number of communications and return for different $\zeta_{\mathrm{th}}$ values", fontsize=14)
    plt.show()


                
        # except KeyboardInterrupt:
        #     train_agents.ask_save()
        #     print("Paused, hit ENTER to continue, type q to quit.")
        #     response = input()
        #     if response == 'q':
        #         break
        #     else:
        #         print('Resuming...')
        #         continue











                


