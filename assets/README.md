The OMX-F URDF is copied from the local ROBOTIS `open_manipulator_description`
package (version 5.1.2), `urdf/omx_f/omx_f.urdf`. It is distributed under the
Apache 2.0 license in this directory. The upstream project is
https://github.com/ROBOTIS-GIT/open_manipulator.

Only kinematics are loaded here; mesh files are not needed. To use a changed
description, set `urdf_path` in your OMX JSON configuration to the generated URDF
(not an unexpanded xacro). The arm has joint1..joint5; the tool frame is
end_effector_link, and gripper_joint_2 mimics gripper_joint_1 with multiplier -1.
