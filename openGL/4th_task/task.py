import pygame
from pygame.locals import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image

verticies = (
    (0, 0, -1),
    (0, 1, -1),
    (1, 1, -1),
    (1, 0, -1),
)


def Cube():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #glPushMatrix()

    glBegin(GL_QUADS)
    for vertex in verticies:
        glTexCoord3d(*vertex)
        glVertex3fv(np.array(vertex))
    glEnd()
    #glPopMatrix()

class Material:
    def __init__(self, filepath: str):
        img = Image.open(filepath).transpose(Image.TRANSPOSE).transpose(Image.ROTATE_90)
        img = img.convert("RGB")
        img_data = np.array(list(img.getdata()), np.int8)
        format = GL_RGB if img.mode == "RGB" else GL_RGBA
        glEnable(GL_TEXTURE_2D)
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0,
                 GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)


    def use(self) -> None:
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)
        
        glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR)
        glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR)


    def disable(self) -> None:
        glDisable(GL_TEXTURE_2D)


    def destroy(self) -> None:
        glDeleteTextures(1, (self.texture,))


def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    rot_cube_map  = { K_UP: (-1.5, 0), K_DOWN: (1.5, 0), K_LEFT: (0, -1.5), K_RIGHT: (0, 1.5)}
    rot_cube = (0, 0)
    texture = Material("openGL/4th_task/ph.jpeg")

    glTranslatef(0.0,0.0, -5)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if event.type == KEYDOWN:
            if event.key in rot_cube_map:
                rot_cube = rot_cube_map[event.key]
            if event.key == K_w:
                glTranslatef(0.0,0.0, -0.5)
            if event.key == K_s:
                glTranslatef(0.0,0.0, 0.5)
        
        if event.type == KEYUP:
            if event.key in rot_cube_map:
                rot_cube = (0, 0)

        glRotatef(rot_cube[1], 0, 1, 0)
        glRotatef(rot_cube[0], 1, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        
        texture.use()
        Cube()
        pygame.display.flip()
        pygame.time.wait(10)


main() 