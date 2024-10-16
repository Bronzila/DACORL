import os

# Define the agents and their corresponding descriptions
agent_descriptions = {
    'combined_e_c': 'exponential decay and constant',
    'combined_e_sg': 'exponential decay and SGDR',
    'combined_e_sg_c': 'exponential decay, SGDR, and constant',
    'combined_e_st': 'exponential decay and step decay',
    'combined_e_st_c': 'exponential decay, step decay, and constant',
    'combined_e_st_sg': 'exponential decay, step decay, and SGDR',
    'combined_sg_c': 'SGDR and constant',
    'combined_st_c': 'step decay and constant',
    'combined_st_sg': 'step decay and SGDR',
    'combined_st_sg_c': 'step decay, SGDR, and constant',
    'combined': 'no special scheduling',
}

# Initialize LaTeX output
latex_code = ""
for function in ["Ackley", "Rastrigin", "Rosenbrock", "Sphere"]:

    # Define the base paths
    base_path = f'figures/experiments/hetero_concat/figures/{function}/'
    comparison_path = f'figures/experiments/hetero_concat/figures/comparison/{function}/'


    for agent, description in agent_descriptions.items():
        # Paths to the images
        behavior_image = os.path.join(base_path, f'{agent}/action_{function}_{agent}_aggregate_60000_aggregate.pdf')
        trajectory_image = os.path.join(comparison_path, f'comparison_{function}_{agent}.pdf')

        # Unique labels for each subfigure
        behavior_label = f'fig:behavior_{function}_{agent}'
        trajectory_label = f'fig:trajectory_{function}_{agent}'

        # Determine the types of scheduling
        if '_e' in agent:
            decay_type = 'exponential decay'
        else:
            decay_type = ''
        
        if 'st' in agent:
            step_type = 'step decay'
        else:
            step_type = ''
        
        if 'sg' in agent:
            sg_type = 'SGDR'
        else:
            sg_type = ''
        
        if '_c' in agent:
            const_type = 'constant'
        else:
            const_type = ''

        # Construct the main caption
        types = [decay_type, step_type, sg_type, const_type]
        types = [t for t in types if t]  # Remove empty strings

        if len(types) > 1:
            main_caption = f'Comparing behavior and function value trajectory on {function} between the {", ".join(types[:-1])} and {types[-1]} teachers and the agent trained on them.'
        else:
            main_caption = f'Comparing behavior and function value trajectory on {function} between all teachers and the agent trained on them.'

        # Generate LaTeX code for each agent
        latex_code += f"""
    \\begin{{figure}}[h]
        \\centering
        \\begin{{subfigure}}[h]{{0.38\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{behavior_image}}}
            \\caption{{Behavior.}}
            \\label{{{behavior_label}}}
        \\end{{subfigure}}
        \\begin{{subfigure}}[h]{{0.38\\textwidth}}
            \\centering
            \\includegraphics[width=\\textwidth]{{{trajectory_image}}}
            \\caption{{Function value trajectory.}}
            \\label{{{trajectory_label}}}
        \\end{{subfigure}}
        \\caption{{{main_caption}}}
        \\label{{fig:comparison_{function}_{agent}}}
    \\end{{figure}}
    """

# Write the output to a file
with open('latex_output.tex', 'w') as file:
    file.write(latex_code)

print("LaTeX code has been generated and saved to 'latex_output.tex'.")
