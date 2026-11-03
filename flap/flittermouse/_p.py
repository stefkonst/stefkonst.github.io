import json
d=json.load(open("data/structure_elec.json"))
pick=["strip 1.5 x 0.2, flat to flap","strip 1.5 x 0.2, on edge","cross 1 x 0.2","cross 1.5 x 0.2","cross 2 x 0.2","tube 1/0.7"]
rows="".join("<tr%s><td>%s</td><td>%.0f</td><td>%.2f</td><td>%.2f</td><td>%.0f</td><td>%.3f</td></tr>\n"%(' class="main"' if r["name"]=="cross 1.5 x 0.2" else "",r["name"].replace(" x "," × "),r["mass_mg"],r["flap_mm"],r["chord_mm"],r["f1_Hz"],r["GJ_mN_m2"]) for n in pick for r in d["le"] if r["name"]==n)
m=d["modes"]
sec=r'''<section class="card" id="structure">
<h2>Leading edge and transmission</h2>
<p>Closed-form beam estimates for one wing at 25 Hz, ±30°. Code: <a href="structure_elec.py">structure_elec.py</a>. Sizes in mm.</p>
<div class="tw"><table>
<tr><th>Leading edge</th><th>Mass (mg)</th><th>Tip bend, flapping (mm)</th><th>Tip bend, skin tension (mm)</th><th>First mode (Hz)</th><th>$GJ$ (mN m²)</th></tr>
'''+rows+r'''</table></div>
<ul class="read">
<li><b>Equation:</b> tip bend $\delta = q b^4 / (8 E I)$, with $q$ the load per length from wing inertia and peak lift, or from chord tension. A flat strip is stiff one way only, so one of the two loads always bends it.</li>
<li><b>Trend:</b> a single strip fails in one direction, so the cross is needed, not optional. The tube is lighter and about 9× stiffer in torsion than the 1.5 mm cross.</li>
<li><b>Trailing-edge anchor:</b> 1 mm off the stroke axis changes the fibre length by about 2 % over the stroke, about the same as the fibre's working stretch. Aim for 0.2 mm or better.</li>
<li><b>Bias spring:</b> about 3.3 mN m/rad from a carbon flexure, for example 0.1 × 2 × 6.6 mm: 2 mg, 0.4 % strain. <b>Hinge:</b> polyimide 12.5 µm × 0.5 mm, 0.65 % strain. At 25 Hz, 2 million cycles take 22 h.</li>
<li><b>Skin modes, no flow:</b> first mode %.0f–%.0f Hz with the batten bonded to the leading edge, and %.0f–%.0f Hz with it hinged, depending on tension. The hinged range includes the flapping frequency.</li>
<li><b>Validate:</b> weigh and load a bonded leading edge; tap test a panel for the first mode at 0 V and at voltage.</li>
</ul>
</section>

<section class="card" id="electrics">
<h2>Electrics</h2>
<p>Scaled from Ren, Kim et al. 2022, Table 1 (6 layers, 35.5 µm, 125 mg, 1.83 nF, 22.1 kΩ). The capacitance formula with $\varepsilon_r = 2.8$ gives %.2f nF for that device, against 1.83 nF in the paper.</p>
<ul class="read">
<li><b>Core, 500 mg, 20 mm long, from 55 mm wide sheets:</b> about %.0f layers at 35.5 µm, %.1f mm outer diameter, %.1f nF, %.0f cm² of electrode. That is %.1f× the electrode area of the 6-layer device, which matters because breakdown probability grows with area.</li>
<li><b>Time constants:</b> about %.0f µs for the core, %.0f µs for the skin and %.0f µs for the fibre band, against a 20 ms half stroke. These assume the sheet resistance implied by that paper and a feed along the span. So electrode RC should not limit the sync, if the electrodes match theirs.</li>
<li><b>Drive, one channel, core + skin + fibre:</b> %.1f nF, %.1f mA, about %.0f mW at 25 Hz if no charge is recovered, against about 14 mW of mechanical output.</li>
<li><b>Validate:</b> measure $R$ and $C$ of each device before any force test, and compare the waveform at the device with the command at 25 Hz.</li>
</ul>
</section>

<section class="card" id="first">'''
t=d["drive_power"]["table"][1]; c=d["core_builds"][1]
sec=sec%(min(m[i]["f_with_air_Hz"][0] for i in range(4)),max(m[i]["f_with_air_Hz"][0] for i in range(4)),min(m[i]["f_with_air_Hz"][0] for i in range(4,8)),max(m[i]["f_with_air_Hz"][0] for i in range(4,8)),d["chen6"]["C_nF_model"],c["layers"],c["od_mm"],c["C_nF"],c["electrode_cm2"],c["ratio_to_chen6"],c["RC_us"],d["skin_electrode"]["RC_us"],d["te_electrode"]["RC_us"],d["drive_power"]["C_nF"],t["I_mA"],t["P_no_recovery_mW"])
s=open("_body.html",encoding="utf-8").read()
s=s.replace('<section class="card" id="first">',sec,1)
s=s.replace('<a href="#first">First pass</a>','<a href="#structure">Structure</a><a href="#electrics">Electrics</a><a href="#first">First pass</a>')
s=s.replace('<li><b>3 Nov, 17:30:</b>','<li><b>3 Nov, 17:20:</b>')
s=s.replace('''not yet validated.</li>
''','''not yet validated.</li>
<li><b>3 Nov, 17:25:</b> leading edge, transmission, skin modes and electrics added. Finite-element runs still going.</li>
''')
open("_body.html","w",encoding="utf-8").write(s)
