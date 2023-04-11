# 498 Project

## Collecting collision data
For collecting collision data from pybullet, use the script `pybullet_experiments/urdf_collision.py`
Running this will start generating `npy` files in `log/`.

### File naming:
`{moving_object_name}_{static_object_name}_{number}.npy`

### File content
```
np.save(filename, {
        'simulation_time': simulation_time,
        'moving_position': moving_position,
        'moving_orientation': moving_orientation,
        'moving_velocity': moving_velocity,
        'moving_angular_velocity': moving_angular_velocity,
        'static_position': static_position,
        'static_orientation': static_orientation,
        'static_velocity': static_velocity,
        'static_angular_velocity': static_angular_velocity,
        'contact_points': contact_points,
        'contact_times': contact_times
    })
``` 