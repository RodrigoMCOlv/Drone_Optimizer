import mujoco
import mujoco.viewer
import numpy as np
import time

xml = """
<mujoco>
    <worldbody>
        <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
        <body pos="0 0 1">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
        </body>
    </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)
for i in range(100):
    mujoco.mj_step(model, data)
    
    if hasattr(viewer, 'user_scn'):
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            np.array([1, 0, 0, 0.5])
        )
        viewer.user_scn.geoms[0].size[0] = 0.05
        viewer.user_scn.geoms[0].pos[:] = np.array([np.sin(i/10.0), 0, 1])
    
    viewer.sync()
    time.sleep(0.02)
viewer.close()
