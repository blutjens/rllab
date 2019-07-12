<%
    
    noise = opts.get("noise", False)
    track_width = 4
    if noise:
        import numpy as np
        track_width += np.random.uniform(-1, 1)

    cart_friction = 0.0005
    pole_friction = 0.000002

    from envs import constants
    ctrl_lim_low = constants.action_low
    ctrl_lim_high = constants.action_high

%>

<box2d>
  <world timestep="0.04"> 
    <body name="weights" type="dynamic" position="0,0.05">
      <fixture density="5" friction="${cart_friction}" shape="polygon" box="0.2, 0.1"/>
    </body>
    <body name="track" type="static" position="0,0.05">
      <fixture friction="${pole_friction}" group="-1" shape="polygon" box="${track_width},0.1"/>
    </body>
    <state type="xpos" body="weights"/> <!-- Height -->
    <state type="xvel" body="weights"/> <!-- Height Rate -->

    <control type="force" body="weights" anchor="0,0" direction="1,0" ctrllimit="${ctrl_lim_low},${ctrl_lim_high}"/>
  </world>
</box2d>
