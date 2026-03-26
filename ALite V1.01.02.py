#!/usr/bin/env python3
import numpy as np
import os
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import random

ascii_art = """
 ▗▄▖ ▗▄▄▖ ▗▄▄▖ ▗▄▄▖  ▗▄▖  ▗▄▖  ▗▄▄▖▗▖ ▗▖▗▄▄▄▖▗▄▄▖ 
▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌ ▐▌▐▌   ▐▌ ▐▌
▐▛▀▜▌▐▛▀▘ ▐▛▀▘ ▐▛▀▚▖▐▌ ▐▌▐▛▀▜▌▐▌   ▐▛▀▜▌▐▛▀▀▘▐▛▀▚▖
▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌▝▚▄▞▘▐▌ ▐▌▝▚▄▄▖▐▌ ▐▌▐▙▄▄▖▐▌ ▐▌
"""

print(ascii_art)


def sum_position_vectors(pqr_file_path):
    rpsi1 = [0.0, 0.0, 0.0]
    atom_count = 0 

    with open(pqr_file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                parts = line.split()
                
                x = float(parts[5])
                y = float(parts[6])
                z = float(parts[7])
                
                rpsi1[0] += x
                rpsi1[1] += y
                rpsi1[2] += z
                
                atom_count += 1

    return rpsi1, atom_count


def translate_to_origin(pqr_file_path, output_file_path, geometric_center):
    with open(pqr_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            if line.startswith("ATOM"):
                parts = line.split()
                
                x = float(parts[5])
                y = float(parts[6])
                z = float(parts[7])
                
                x -= geometric_center[0]
                y -= geometric_center[1]
                z -= geometric_center[2]
                
                parts[5] = f"{x:.3f}"
                parts[6] = f"{y:.3f}"
                parts[7] = f"{z:.3f}"
                
                new_line = "{: <6s} {: >5s} {: <4s}  {: >3s} {: >4s} {: >8s} {: >8s} {: >8s} {: >8s} {: >8s}\n".format(
                    parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7], parts[8], parts[9]
                )
                
                outfile.write(new_line)
            else:
                outfile.write(line)


def translate_to_target(pqr_file_path, output_file_path, geometric_center, target_vector):
    with open(pqr_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            if line.startswith("ATOM"):
                parts = line.split()
                
                x = float(parts[5])
                y = float(parts[6])
                z = float(parts[7])
                
                x += target_vector[0] - geometric_center[0]
                y += target_vector[1] - geometric_center[1]
                z += target_vector[2] - geometric_center[2]
                
                parts[5] = f"{x:.3f}"
                parts[6] = f"{y:.3f}"
                parts[7] = f"{z:.3f}"
                
                new_line = "{: <6s} {: >5s} {: <4s}  {: >3s} {: >4s} {: >8s} {: >8s} {: >8s} {: >8s} {: >8s}\n".format(
                    parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7], parts[8], parts[9]
                )
                
                outfile.write(new_line)
            else:
                outfile.write(line)

def CSwriter(pqr_file1_path, pqr_file2_path, output_file_path):
    with open(output_file_path, 'w') as outfile:
        with open(pqr_file1_path, 'r') as infile:
            for line in infile:
                if line.startswith("ATOM"):
                    parts = line.split()
                    formatted_line = "{: <6s} {: >5s} {: <4s} {: >3s} {: >4s} {: >8s} {: >8s} {: >8s} {: >8s} {: >8s}\n".format(
                        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7], parts[8], parts[9]
                    )
                    outfile.write(formatted_line)
        
        with open(pqr_file2_path, 'r') as infile:
            for line in infile:
                if line.startswith("ATOM"):
                    parts = line.split()
                    formatted_line = "{: <6s} {: >5s} {: <4s} {: >3s} {: >4s} {: >8s} {: >8s} {: >8s} {: >8s} {: >8s}\n".format(
                        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7], parts[8], parts[9]
                    )
                    outfile.write(formatted_line)


def balthazar(apbs_path, input_file, pqr_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    if not os.path.exists(pqr_file):
        raise FileNotFoundError(f"PQR file '{pqr_file}' not found.")
    
    command = [apbs_path, input_file]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("APBS calculation completed successfully.")
        print("Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error during APBS execution:")
        print(e.stderr)


if __name__ == "__main__":
    final_proteins_array = []
    steps = 4
    n = 1
    T_min = 55
    T_max = 100

    for T in np.arange (T_min, T_max, ((T_max - T_min)/ (steps + 1))):
        p = int(100 * (n - 1)/ steps)
        print( p ,'%')

        pqr_file = f"rotated_protein.pqr"
        output_file = f"final_protein_{n}.pqr" 
        final_proteins_array.append(output_file)
        
        target_vector = [0.0, 0.0, T]

        rpsi1, atom_count = sum_position_vectors(pqr_file)
        geometric_center = [
            rpsi1[0] / atom_count,
            rpsi1[1] / atom_count,
            rpsi1[2] / atom_count
        ]
        
        translate_to_target(pqr_file, output_file, geometric_center, target_vector)
        
        pqr_file1 = f"final_protein_{n}.pqr"
        pqr_file2 = f"final_membrane.pqr"
        transformed_pqr = f"CS2_{T}.pqr"
        
        CSwriter(pqr_file1, pqr_file2, transformed_pqr)
        
        n += 1                
    

phrase = ['Do not unsheathe me without reason, do not wield me without valor','May all your dreams come true','All this love was once anger', 'All this anger was once love', 'It was too harsh, you were different. But despite everything it\'s still you', 'Together we go forward', 'I believe in the good nature of people, in spite of everything i experienced', 'Ultimately i have to be brave', 'It gets better, it always does', 'The butterfly you chased was meant to show you the beauty of letting go', 'Enjoy the butterflies', 'It\'s okay to make mistakes', 'Goodbye, my friend', 'Maybe it\'s not for me', 'I hate you for what you did, and i miss you like a little kid', 'The sun will continue to shine on earth for billions of years to come', 'Do it for the nights where you only had yourself', 'Courage is not the absence of fear', 'For as long as i exist, you will always be loved', 'Nothing in life scares me, except the things i want the most', 'What are you trying to accomplish by expecting things from others?', 'Life', 'You are not required to set yourself on fire to keep other people warm', 'Today, i choose to live', 'How long must i try?', 'We really did have everything, didnt we?', 'I am proud of helping you to make something pretty', 'He just wanted someone to be proud', 'The pain is tremendous, but so is the beauty', 'Have you ever seen the rain glow?', 'You\'ve mastered surviving, it\'s time to live now', 'Failure is not an option']

RI = random.randint(0,len(phrase)-1)

print(phrase[RI])























