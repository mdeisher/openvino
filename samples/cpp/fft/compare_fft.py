import subprocess

fft_sizes = ( 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 )

for N in fft_sizes:
    in_csv = "fft_input"+str(N)+".csv"
    out_csv = "fft_output"+str(N)+".csv"
    ir_xml = "fft"+str(N)+".xml"
    ir_bin = "fft"+str(N)+".bin"
    in_ark = "fft_input"+str(N)+".ark"
    ref_ark = "fft_output"+str(N)+".ark"
    out_ark = "fft_output.ark"
    subprocess.run(".\Debug\make_fft.exe "+str(N), shell=True)
    subprocess.run("move fft_input.csv "+in_csv, shell=True)
    subprocess.run("move fft_output.csv "+out_csv, shell=True)
    subprocess.run("move fft.xml "+ir_xml, shell=True)
    subprocess.run("move fft.bin "+ir_bin, shell=True)
    subprocess.run("python csvtoark.py "+in_csv, shell=True)
    subprocess.run("python csvtoark.py "+out_csv, shell=True)
    print(".\Debug\speech_sample.exe -m "+ir_xml+" -i "+in_ark+" -r "+ref_ark+" -o "+out_ark)
    subprocess.run(".\Debug\speech_sample.exe -m "+ir_xml+" -i "+in_ark+" -r "+ref_ark+" -o "+out_ark, shell=True)
    subprocess.run("python plotark.py "+out_ark+" "+ref_ark, shell=True)

