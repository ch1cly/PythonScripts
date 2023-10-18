import pygame
import random
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from random import randint as ri

'''
На примере готового кубик-рубика рассмотреть обработку клавиш для вращения граней,
прорисовку quad как массива кубиков, заливку граней разными цветами.

*Добавить клавиши для вращения кубик-рубика размером 4*4.
'''


vertices = (
    ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1), (-1, -1, -1),
    ( 1, -1,  1), ( 1,  1,  1), (-1, -1,  1), (-1,  1,  1)
)
edges = ((0,1),(0,3),(0,4),(2,1),(2,3),(2,7),(6,3),(6,4),(6,7),(5,1),(5,4),(5,7))
surfaces = ((0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4), (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6))
colors = ((1, 0, 0), (0, 1, 0), (1, 0.5, 0), (1, 1, 0), (1, 1, 1), (0, 0, 1))

class Cube():
    def __init__(self, id, N, scale):
        self.N = N
        self.scale = scale
        self.init_i = [*id]
        self.current_i = [*id]
        self.rot = [[1 if i==j else 0 for i in range(3)] for j in range(3)]

    def isAffected(self, axis, slice, dir):
        return self.current_i[axis] == slice

    def update(self, axis, slice, dir):

        if not self.isAffected(axis, slice, dir):
            return

        i, j = (axis+1) % 3, (axis+2) % 3
        for k in range(3):
            self.rot[k][i], self.rot[k][j] = -self.rot[k][j]*dir, self.rot[k][i]*dir

        self.current_i[i], self.current_i[j] = (
            self.current_i[j] if dir < 0 else self.N - 1 - self.current_i[j],
            self.current_i[i] if dir > 0 else self.N - 1 - self.current_i[i] )

    def transformMat(self):
        scaleA = [[s*self.scale for s in a] for a in self.rot]  
        scaleT = [(p-(self.N-1)/2)*2.1*self.scale for p in self.current_i] 
        return [*scaleA[0], 0, *scaleA[1], 0, *scaleA[2], 0, *scaleT, 1]

    def draw(self, col, surf, vert, animate, angle, axis, slice, dir):

        glPushMatrix()
        if animate and self.isAffected(axis, slice, dir):
            glRotatef( angle*dir, *[1 if i==axis else 0 for i in range(3)] )
        glMultMatrixf( self.transformMat() )

        glBegin(GL_QUADS)
        for i in range(len(surf)):
            glColor3fv(colors[i])
            for j in surf[i]:
                glVertex3fv(vertices[j])
        glEnd()

        glLineWidth(3.0)
        for i in range(len(surf)):
            glBegin(GL_LINE_LOOP)
            glColor3fv((0.0,0.0,0.0))
            for j in surf[i]:
                glVertex3fv(vertices[j])
            glEnd()
        
        glPopMatrix()


class EntireCube():
    def __init__(self, N, scale):
        self.N = N
        cr = range(self.N)
        self.cubes = [Cube((x, y, z), self.N, scale) for x in cr for y in cr for z in cr]

    def mainloop(self):

        
        rot_cube_map  = { K_UP: (-1, 0), K_DOWN: (1, 0), K_LEFT: (0, -1), K_RIGHT: (0, 1)}
        if self.N == 3:
            rot_slice_map = {
                K_1: (0, 0, 1),  K_q: (0, 1, 1),  K_a: (0, 2, 1), 
                K_2: (0, 0, -1), K_w: (0, 1, -1), K_s: (0, 2, -1),
                K_3: (1, 0, 1),  K_e: (1, 1, 1),  K_d: (1, 2, 1), 
                K_4: (1, 0, -1), K_r: (1, 1, -1), K_f: (1, 2, -1),
                K_5: (2, 0, 1),  K_t: (2, 1, 1),  K_g: (2, 2, 1),  
                K_6: (2, 0, -1), K_y: (2, 1, -1), K_h: (2, 2, -1)
            }
            #delete, not for homeWork! 
            kkeys = (K_1, K_q, K_a, K_2, K_w, K_s, K_3, K_e,
                     K_d, K_4, K_r, K_f, K_5, K_t, K_g, K_6, 
                     K_y,K_h)  
        elif self.N == 4:
            rot_slice_map = {
                K_1: (0, 0, 1),  K_q: (0, 1, 1),  K_a: (0, 2, 1),  K_z: (0, 3, 1),
                K_2: (0, 0, -1), K_w: (0, 1, -1), K_s: (0, 2, -1), K_x: (0, 3, -1),
                K_3: (1, 0, 1),  K_e: (1, 1, 1),  K_d: (1, 2, 1),  K_c: (1, 3, 1),
                K_4: (1, 0, -1), K_r: (1, 1, -1), K_f: (1, 2, -1), K_v: (1, 3, -1),
                K_5: (2, 0, 1),  K_t: (2, 1, 1),  K_g: (2, 2, 1),  K_b: (2, 3, 1),
                K_6: (2, 0, -1), K_y: (2, 1, -1), K_h: (2, 2, -1), K_n: (2, 3, -1)
            }
            #delete, not for homeWork! 
            kkeys = (K_1, K_q, K_a, K_z, K_2, K_w, K_s, K_x, K_3, K_e, 
                     K_d, K_c, K_4, K_r, K_f, K_v, K_5, K_t, K_g, K_b, 
                     K_6, K_y,K_h, K_n)
            

        
        ang_x, ang_y, rot_cube = 0, 0, (0, 0)
        animate, animate_ang, animate_speed = False, 0, 5
        action = (0, 0, 0)
        rot_memory_stack = [] #added stack to remember actions
        isK_kPressed = False #flag Added
        while True:
            if isK_kPressed: #if glag 
                if len(rot_memory_stack) > 0 and not animate: #if not rotated
                    temp_rot = list(rot_slice_map[rot_memory_stack[-1]]) #take button
                    rot_memory_stack.pop()
                    temp_rot[-1] = temp_rot[-1]*-1
                    animate, action = True, tuple(temp_rot) #animate
                if len(rot_memory_stack) == 0: #no things to animate
                    isK_kPressed = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == KEYDOWN:
                    if event.key in rot_cube_map:
                        rot_cube = rot_cube_map[event.key]

                    if not animate and event.key == K_k: #if 'k' is pressed
                        isK_kPressed = True
                        
                    if not animate and not isK_kPressed \
                        and (event.key in rot_slice_map or event.key == K_o) :
                        if event.key != K_o:                  #delete, not for homeWork! 
                            t_key = event.key
                        else:                                 #delete, not for homeWork! 
                            t_key = kkeys[ri(0,len(kkeys)-1)] #delete, not for homeWork!  

                        animate, action = True, rot_slice_map[t_key]
                        flag = True

                        if len(rot_memory_stack) > 0:       
                            #delete, not for homeWork! 
                            t_key1,t_key2 = list(rot_slice_map[t_key]),list(rot_slice_map[rot_memory_stack[-1]])  #delete, not for homeWork! 
                            t_key1[-1] = t_key1[-1]*-1      #delete, not for homeWork! 
                            flag = not t_key1 == t_key2     #delete, not for homeWork! 
                        if flag:
                            rot_memory_stack.append(t_key) #remember key
                        else:                               #delete, not for homeWork! 
                            rot_memory_stack.pop()          #delete, not for homeWork! 
                if event.type == KEYUP:
                    if event.key in rot_cube_map:
                        rot_cube = (0, 0)

            ang_x += rot_cube[0]*2
            ang_y += rot_cube[1]*2

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0, 0, -40)
            glRotatef(ang_y, 0, 1, 0)
            glRotatef(ang_x, 1, 0, 0)

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

            if animate:
                if animate_ang >= 90:
                    for cube in self.cubes:
                        cube.update(*action)
                    animate, animate_ang = False, 0

            for cube in self.cubes:
                cube.draw(colors, surfaces, vertices, animate, animate_ang, *action)
            if animate:
                animate_ang += animate_speed

            pygame.display.flip()
            pygame.time.wait(10)

def main():

    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    glEnable(GL_DEPTH_TEST) 

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    #first parameter is shape of cube!!!
    NewEntireCube = EntireCube(4, 1.5)
    NewEntireCube.mainloop()

if __name__ == '__main__':
    main()
    pygame.quit()
    quit()