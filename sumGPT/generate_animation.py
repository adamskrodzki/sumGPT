import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# Define the character vocabulary
CHAR_VOCAB = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '=', '\n', '_', 'X', 'Y']

# Function to convert logit to color
def logit_to_color(logit):
    prob = math.exp(logit)
    normalized = -255 / (-1 + prob)
    return plt.cm.Greens(normalized)

# Read and parse the data
def parse_data(file_path):
    queries = []
    logits = []
    current_query = None
    current_logits = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Query:'):
                if current_query is not None:
                    queries.append(current_query)
                    logits.append(current_logits)
                current_query = line.split('Query: ')[1].strip()
                diff = current_query.split('=')[-1].strip()
                current_query = current_query.replace('=', f'{diff}=')
                current_logits = []
            elif 'Logit:' in line:
                data = line.split('Logit: ')[1].split(', Diff:')[0].strip()
                logit_values = eval(data)
                current_logits.append(logit_values)
        if current_query is not None:
            queries.append(current_query)
            logits.append(current_logits)
    return queries, logits
import matplotlib.pyplot as plt
import numpy as np

def logit_to_color2(logit):
    # Dummy function to convert logit to color
    return 'blue' if logit > 0 else 'red'

def create_frames(queries, logits):
    frames = []
    for i, query in enumerate(queries):
        fig, ax = plt.subplots()
        ax.set_title(f'Query: {query}')
        ax.axis('off')
        
        num_rows = len(logits[i])
        num_cols = len(logits[i][0])
        
        ax.set_xlim(0, num_cols)
        ax.set_ylim(-num_rows, 0)
        
        for j, logit_row in enumerate(logits[i]):
            for k, logit in enumerate(logit_row):
                color = logit_to_color(logit)
                ax.add_patch(plt.Rectangle((k, -j - 1), 1, 1, color=color))
                ax.text(k + 0.5, -j - 0.5, CHAR_VOCAB[k], ha='center', va='center', fontsize=8)
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)
    return frames

# Animate and save frames
def save_animation(frames, file_name, fps=2):
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save(file_name, fps=fps, writer='ffmpeg')

# Main function
def main(file_path, output_file):
    queries, logits = parse_data(file_path)
    frames = create_frames(queries, logits)
    save_animation(frames, output_file)

# Run the main function
main('visualisation.txt', 'animation.mp4')  # Change to .mp4 for MP4 output
