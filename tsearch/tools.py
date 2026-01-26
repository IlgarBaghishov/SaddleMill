def save_ordered_traj_names(all_traj_files):
    with open('traj_files_ordered.txt', 'w') as f:
        for name in all_traj_files:
            f.write(f"{name}\n")

