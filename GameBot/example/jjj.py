v0 = 4        # линейная скорость
N = 1000      # количество бактерий
Epochs =  500 # количество шагов
L    = 300    # размер области
R    = 5      # радиус взаимодействия
observation_R = 2*R # Радиус видимости соседей

fig = plt.figure()
camera = Camera(fig)
random.seed(123)
theCure = Cure()
observation = theCure.reset()

# информационная плашка
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
sum_reward = 0
for i in range(200):
    action = np.sum(actor.predict(observation.reshape((1,1,5))))# % (2*np.pi)
    observation, reward, done, _ = theCure.step(action)
    sum_reward += reward
    if done:
      print('Победа  на шаге',i, ' захвачено ',observation[0]*20,'бактерий. Награда ',sum_reward)
      break
    # покажем бактерий
    bacteria_x,bacteria_y = theCure.get_bacteria()
    plt.scatter(bacteria_x, bacteria_y, c='red')    #  метод, отображающий данные в виде точек
    # покажем робота
    x, y, r = theCure.get_position()
    plt.scatter(x, y, c='blue')
    fig = plt.gcf()
    ax = fig.gca()
    circle = plt.Circle((x, y), r, color='b', fill=False)
    ax.add_patch(circle)

    textstr = '\n'.join((
    r'epoch=%d' % (i, ),
    r'points=%d' % (reward, ),
    ))

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
      verticalalignment='top', bbox=props)

    camera.snap()

print('Итоговое вознаграждение',sum_reward)
theCure.close()
animation = camera.animate()
#animation.save('celluloid_minimal.gif', writer = 'imagemagick')
animation