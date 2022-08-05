from PIL import Image
import imageio
import os


def main():
    path = [f"./{i}" for i in os.listdir("./zxc")]
    path = sorted(os.listdir("./zxc/"))[1:]
    
    imgs = [Image.open("./zxc/" + i) for i in path]
    imageio.mimsave('./result.gif', imgs, fps=3)


if __name__ == '__main__':
    main()