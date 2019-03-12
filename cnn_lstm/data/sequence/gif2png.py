from PIL import Image
import sys, os

def processImage(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load", infile)
        sys.exit(1)
    i = 0
    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            new_im.save(os.path.join('tst_gif', 'foo'+str(i)+'.png'))

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence

processImage('test_gif.gif')
