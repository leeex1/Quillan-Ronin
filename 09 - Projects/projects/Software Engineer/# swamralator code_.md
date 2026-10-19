---
file_type: reference
domain: dev
status: active
tags: [code, dashboard, swarmalator]
---
# swamralator code:

```html
<body style="height: 100%; display: flex; flex-direction: column; overflow: hidden; background-color: var(--surface-container); margin: 0px;"><script>
/**
 * Swarmalator Thermodynamic Telemetry Dashboard
 * Visualizing the tradeoff between Phase-Lock Time and E_ICE Load.
 */

// 1. Setup App (UI + State)
const { state, ui } = WH.createApp({
    title: "Swarmalator Thermodynamic Telemetry",
    params: {
        couplingK: { value: 0.5, min: 0.1, max: 1.0, step: 0.05, label: "Coupling Constant (K)" },
        enableResets: { value: false, label: "Enable Kinetic Resets" },
        regenerate: { type: 'button', label: 'Randomize Runs', onClick: (s) => generateRuns(s) }
    }
});

// 2. Initialize Internal State
state.runs = [];
const CRITICAL_LIMIT = 2.8;

// Data Generation Logic
function generateRuns(s) {
    const types = ['Temporal Fold', 'Causal Loop', 'Entropic Drift', 'Quantum Lock'];
    const newRuns = [];
    for (let i = 0; i < 100; i++) {
        newRuns.push({
            id: `PX-${1000 + i}`,
            type: types[Math.floor(Math.random() * types.length)],
            // Base values that will be modified by K
            baseX: Math.random() * 3000,
            baseY: 1.0 + Math.random() * 2.5,
            jitter: Math.random()
        });
    }
    state.runs = newRuns;
    updateHUD();
}

// Stats Calculation for HUD
function updateHUD() {
    if (!state.runs.length) return;
    
    const K = state.couplingK;
    let totalTime = 0;
    let successes = 0;
    let terminated = 0;

    state.runs.forEach(d => {
        // Higher K shifts points left (lower time) but higher (higher load)
        const x = d.baseX * (1.1 - K);
        const y = d.baseY * (0.5 + K);
        
        totalTime += x;
        if (y < CRITICAL_LIMIT) {
            successes++;
        } else {
            terminated++;
        }
    });

    ui.setHUD([
        { label: "Avg Phase-Lock", value: `${(totalTime / state.runs.length).toFixed(0)} ms` },
        { label: "Success Rate", value: `${((successes / state.runs.length) * 100).toFixed(1)}%` },
        { label: "Runs Terminated", value: terminated.toString() }
    ]);
}

// 3. Initialize D3 Visualization
const render = WH.initD3('viz', (selection, width, height) => {
    const margin = { top: 40, right: 40, bottom: 60, left: 70 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const svg = selection.append('svg')
        .attr('width', width)
        .attr('height', height);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear().domain([0, 3000]).range([0, innerWidth]);
    const yScale = d3.scaleLinear().domain([1.0, 3.5]).range([innerHeight, 0]);

    // Axes
    const xAxis = g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(5));
    
    const yAxis = g.append('g')
        .call(d3.axisLeft(yScale));

    // Labels
    g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 45)
        .attr('fill', WH.getColor('--on-surface-default'))
        .attr('text-anchor', 'middle')
        .text('Phase-Lock Time (ms)');

    g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -50)
        .attr('fill', WH.getColor('--on-surface-default'))
        .attr('text-anchor', 'middle')
        .text('E_ICE Load (Joules x 10⁻⁸)');

    // Critical Limit Line
    g.append('line')
        .attr('x1', 0)
        .attr('x2', innerWidth)
        .attr('y1', yScale(CRITICAL_LIMIT))
        .attr('y2', yScale(CRITICAL_LIMIT))
        .attr('stroke', WH.getColor('--negative'))
        .attr('stroke-dasharray', '5,5')
        .attr('stroke-width', 2);

    g.append('text')
        .attr('x', innerWidth - 5)
        .attr('y', yScale(CRITICAL_LIMIT) - 10)
        .attr('text-anchor', 'end')
        .attr('fill', WH.getColor('--negative'))
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text('CRITICAL E_ICE LIMIT');

    // Scatter container
    const pointsG = g.append('g');

    // The Update Function
    return () => {
        const K = state.couplingK;
        const resetsEnabled = state.enableResets;

        const data = state.runs.map(d => {
            let x = d.baseX * (1.1 - K);
            let y = d.baseY * (0.5 + K);
            
            // Jitter for resets
            if (resetsEnabled) {
                x += (d.jitter - 0.5) * 200 * K;
                y += (d.jitter - 0.5) * 0.2 * K;
            }

            // Classification
            let category = 'Success';
            let color = '--chart-2'; // Green
            
            if (y >= CRITICAL_LIMIT) {
                if (x < 1500) {
                    category = 'Kinetic Reset';
                    color = '--chart-3'; // Yellow/Orange
                } else {
                    category = 'Semantic Death Spiral';
                    color = '--chart-4'; // Red
                }
            }

            return { ...d, x, y, category, color };
        });

        const circles = pointsG.selectAll('circle')
            .data(data, d => d.id);

        // Enter
        circles.enter()
            .append('circle')
            .attr('r', 6)
            .attr('fill', d => WH.getColor(d.color))
            .attr('stroke', WH.getColor('--surface'))
            .attr('stroke-width', 1)
            .attr('opacity', 0.8)
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .append('title')
            .text(d => `ID: ${d.id}\nType: ${d.type}\nStatus: ${d.category}\nTime: ${d.x.toFixed(0)}ms\nEnergy: ${d.y.toFixed(2)} J`);

        // Update
        circles.transition()
            .duration(500)
            .ease(d3.easeQuadOut)
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('fill', d => WH.getColor(d.color));

        circles.select('title')
            .text(d => `ID: ${d.id}\nType: ${d.type}\nStatus: ${d.category}\nTime: ${d.x.toFixed(0)}ms\nEnergy: ${d.y.toFixed(2)} J`);

        // Exit
        circles.exit().remove();
        
        updateHUD();
    };
});

// 4. Connect State -> View
state._subscribe(() => {
    render();
});

// Start Simulation
generateRuns(state);
render();

</script><div class="widget-header"><div class="header-top"><h3 class="widget-title" style="font-family: &quot;Google Sans&quot;, sans-serif; font-size: 16px; line-height: 24px; font-weight: 500; letter-spacing: 0px;">Swarmalator Thermodynamic Telemetry</h3><div id="widget-dashboard" class="widget-dashboard"><div class="dash-pill"><span class="dash-label" style="font-family: &quot;Google Sans&quot;, sans-serif; font-size: 11px; line-height: 16px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;">AVG PHASE-LOCK</span><span class="dash-value" style="color: var(--on-surface-default); font-family: var(--ff-mono, &quot;Google Code&quot;), &quot;SF Mono&quot;, &quot;Roboto Mono&quot;, monospace; font-size: 14px; line-height: 20px; font-weight: 700; letter-spacing: 0px;">159 ms</span></div><div class="dash-pill"><span class="dash-label" style="font-family: &quot;Google Sans&quot;, sans-serif; font-size: 11px; line-height: 16px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;">SUCCESS RATE</span><span class="dash-value" style="color: var(--on-surface-default); font-family: var(--ff-mono, &quot;Google Code&quot;), &quot;SF Mono&quot;, &quot;Roboto Mono&quot;, monospace; font-size: 14px; line-height: 20px; font-weight: 700; letter-spacing: 0px;">37.0%</span></div><div class="dash-pill"><span class="dash-label" style="font-family: &quot;Google Sans&quot;, sans-serif; font-size: 11px; line-height: 16px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;">RUNS TERMINATED</span><span class="dash-value" style="color: var(--on-surface-default); font-family: var(--ff-mono, &quot;Google Code&quot;), &quot;SF Mono&quot;, &quot;Roboto Mono&quot;, monospace; font-size: 14px; line-height: 20px; font-weight: 700; letter-spacing: 0px;">63</span></div></div></div><div id="app-status" class="header-status" style="display: none; font-family: &quot;Google Sans&quot;, sans-serif; font-size: 11px; line-height: 16px; font-weight: 500; letter-spacing: 0px;"></div></div><div id="viz" class="widget-ui-part viz-container grow bg-surface-container relative overflow-hidden"><div id="viz-badges" class="viz-badges"></div><svg width="100%" height="100%" viewBox="0,0,651.6666259765625,578.454833984375"><g transform="translate(70,40)"><g transform="translate(0,478.454833984375)" fill="none" font-size="10" font-family="sans-serif" text-anchor="middle"><path class="domain" stroke="currentColor" d="M0,6V0H541.6666259765625V6"></path><g class="tick" opacity="1" transform="translate(0,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">0</text></g><g class="tick" opacity="1" transform="translate(90.27777099609375,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">500</text></g><g class="tick" opacity="1" transform="translate(180.5555419921875,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">1,000</text></g><g class="tick" opacity="1" transform="translate(270.83331298828125,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">1,500</text></g><g class="tick" opacity="1" transform="translate(361.111083984375,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">2,000</text></g><g class="tick" opacity="1" transform="translate(451.38885498046875,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">2,500</text></g><g class="tick" opacity="1" transform="translate(541.6666259765625,0)"><line stroke="currentColor" y2="6"></line><text fill="currentColor" y="9" dy="0.71em">3,000</text></g></g><g fill="none" font-size="10" font-family="sans-serif" text-anchor="end"><path class="domain" stroke="currentColor" d="M-6,478.454833984375H0V0H-6"></path><g class="tick" opacity="1" transform="translate(0,478.454833984375)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">1.0</text></g><g class="tick" opacity="1" transform="translate(0,440.17844726562504)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">1.2</text></g><g class="tick" opacity="1" transform="translate(0,401.902060546875)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">1.4</text></g><g class="tick" opacity="1" transform="translate(0,363.625673828125)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">1.6</text></g><g class="tick" opacity="1" transform="translate(0,325.349287109375)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">1.8</text></g><g class="tick" opacity="1" transform="translate(0,287.07290039062497)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">2.0</text></g><g class="tick" opacity="1" transform="translate(0,248.79651367187495)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">2.2</text></g><g class="tick" opacity="1" transform="translate(0,210.52012695312501)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">2.4</text></g><g class="tick" opacity="1" transform="translate(0,172.243740234375)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">2.6</text></g><g class="tick" opacity="1" transform="translate(0,133.967353515625)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">2.8</text></g><g class="tick" opacity="1" transform="translate(0,95.69096679687497)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">3.0</text></g><g class="tick" opacity="1" transform="translate(0,57.41458007812494)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">3.2</text></g><g class="tick" opacity="1" transform="translate(0,19.138193359375016)"><line stroke="currentColor" x2="-6"></line><text fill="currentColor" x="-9" dy="0.32em">3.4</text></g></g><text x="270.83331298828125" y="523.454833984375" fill="#FFFFFF" text-anchor="middle">Phase-Lock Time (ms)</text><text transform="rotate(-90)" x="-239.2274169921875" y="-50" fill="#FFFFFF" text-anchor="middle">E_ICE Load (Joules x 10⁻⁸)</text><line x1="0" x2="541.6666259765625" y1="133.967353515625" y2="133.967353515625" stroke="#F2B8B5" stroke-dasharray="5,5" stroke-width="2"></line><text x="536.6666259765625" y="123.967353515625" text-anchor="end" fill="#F2B8B5" style="font-size: 12px; font-weight: bold;">CRITICAL E_ICE LIMIT</text><g><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="-2.972779839458494" cy="55.933620200200956"><title>ID: PX-1000
Type: Entropic Drift
Status: Kinetic Reset
Time: -16ms
Energy: 3.21 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="21.313567209749703" cy="203.22031091867134"><title>ID: PX-1001
Type: Quantum Lock
Status: Success
Time: 118ms
Energy: 2.44 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="33.24569967363398" cy="109.3937269978754"><title>ID: PX-1002
Type: Causal Loop
Status: Kinetic Reset
Time: 184ms
Energy: 2.93 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="45.95703978754582" cy="-70.6621390924907"><title>ID: PX-1003
Type: Entropic Drift
Status: Kinetic Reset
Time: 255ms
Energy: 3.87 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="63.24135970802318" cy="-177.1402423387781"><title>ID: PX-1004
Type: Quantum Lock
Status: Kinetic Reset
Time: 350ms
Energy: 4.43 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="51.5688897120698" cy="-53.255867748362654"><title>ID: PX-1005
Type: Quantum Lock
Status: Kinetic Reset
Time: 286ms
Energy: 3.78 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="64.46885642183001" cy="247.0623680242582"><title>ID: PX-1006
Type: Entropic Drift
Status: Success
Time: 357ms
Energy: 2.21 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="53.79049294742199" cy="219.54774228069166"><title>ID: PX-1007
Type: Causal Loop
Status: Success
Time: 298ms
Energy: 2.35 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="34.792979614803315" cy="110.31725561649203"><title>ID: PX-1008
Type: Quantum Lock
Status: Kinetic Reset
Time: 193ms
Energy: 2.92 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="40.8839682071259" cy="-214.61353524360737"><title>ID: PX-1009
Type: Quantum Lock
Status: Kinetic Reset
Time: 226ms
Energy: 4.62 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="1.9760894575173107" cy="-277.89149550796293"><title>ID: PX-1010
Type: Quantum Lock
Status: Kinetic Reset
Time: 11ms
Energy: 4.95 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="39.609005553456214" cy="171.86719785164976"><title>ID: PX-1011
Type: Quantum Lock
Status: Success
Time: 219ms
Energy: 2.60 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="3.966232683791191" cy="347.3503018455599"><title>ID: PX-1012
Type: Quantum Lock
Status: Success
Time: 22ms
Energy: 1.69 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="-9.24148821038468" cy="301.10874211219243"><title>ID: PX-1013
Type: Quantum Lock
Status: Success
Time: -51ms
Energy: 1.93 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="69.0627482800977" cy="331.92078525769415"><title>ID: PX-1014
Type: Temporal Fold
Status: Success
Time: 383ms
Energy: 1.77 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="-6.788667928417008" cy="-41.12226606969556"><title>ID: PX-1015
Type: Causal Loop
Status: Kinetic Reset
Time: -38ms
Energy: 3.71 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="56.26982174641221" cy="-77.7018229873589"><title>ID: PX-1016
Type: Causal Loop
Status: Kinetic Reset
Time: 312ms
Energy: 3.91 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="26.538072730719136" cy="-260.90655185049724"><title>ID: PX-1017
Type: Entropic Drift
Status: Kinetic Reset
Time: 147ms
Energy: 4.86 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="40.506558119919724" cy="-49.45791155216478"><title>ID: PX-1018
Type: Quantum Lock
Status: Kinetic Reset
Time: 224ms
Energy: 3.76 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="45.52306530272623" cy="-31.13561334084942"><title>ID: PX-1019
Type: Entropic Drift
Status: Kinetic Reset
Time: 252ms
Energy: 3.66 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="50.796101777788934" cy="-334.60799737832866"><title>ID: PX-1020
Type: Temporal Fold
Status: Kinetic Reset
Time: 281ms
Energy: 5.25 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="42.38161329531139" cy="-74.25461452007343"><title>ID: PX-1021
Type: Causal Loop
Status: Kinetic Reset
Time: 235ms
Energy: 3.89 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="11.680569437323616" cy="34.58993845971704"><title>ID: PX-1022
Type: Entropic Drift
Status: Kinetic Reset
Time: 65ms
Energy: 3.32 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="18.515729860821285" cy="86.01634235806907"><title>ID: PX-1023
Type: Temporal Fold
Status: Kinetic Reset
Time: 103ms
Energy: 3.05 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="13.465887669516277" cy="374.13839327123407"><title>ID: PX-1024
Type: Temporal Fold
Status: Success
Time: 75ms
Energy: 1.55 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="52.64085420184855" cy="47.52030947570576"><title>ID: PX-1025
Type: Entropic Drift
Status: Kinetic Reset
Time: 292ms
Energy: 3.25 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="-10.68611901939335" cy="-92.17737798143953"><title>ID: PX-1026
Type: Entropic Drift
Status: Kinetic Reset
Time: -59ms
Energy: 3.98 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="57.225430555878404" cy="-104.37797276284881"><title>ID: PX-1027
Type: Causal Loop
Status: Kinetic Reset
Time: 317ms
Energy: 4.05 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="25.93376935488339" cy="-238.72308208672266"><title>ID: PX-1028
Type: Quantum Lock
Status: Kinetic Reset
Time: 144ms
Energy: 4.75 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="19.157581579095545" cy="128.32588883306815"><title>ID: PX-1029
Type: Entropic Drift
Status: Kinetic Reset
Time: 106ms
Energy: 2.83 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="33.83063619308942" cy="-27.25772907996582"><title>ID: PX-1030
Type: Entropic Drift
Status: Kinetic Reset
Time: 187ms
Energy: 3.64 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="47.502320220485316" cy="257.629965878208"><title>ID: PX-1031
Type: Quantum Lock
Status: Success
Time: 263ms
Energy: 2.15 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="24.30428537828826" cy="39.72728694445089"><title>ID: PX-1032
Type: Causal Loop
Status: Kinetic Reset
Time: 135ms
Energy: 3.29 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="19.817472239965927" cy="349.7359357758631"><title>ID: PX-1033
Type: Quantum Lock
Status: Success
Time: 110ms
Energy: 1.67 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="14.671477910002586" cy="-186.77633716694174"><title>ID: PX-1034
Type: Causal Loop
Status: Kinetic Reset
Time: 81ms
Energy: 4.48 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="33.3040387269664" cy="23.18553903480713"><title>ID: PX-1035
Type: Quantum Lock
Status: Kinetic Reset
Time: 184ms
Energy: 3.38 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="20.36011287423484" cy="-303.32994509112075"><title>ID: PX-1036
Type: Quantum Lock
Status: Kinetic Reset
Time: 113ms
Energy: 5.08 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="54.27490114063541" cy="217.23572821424398"><title>ID: PX-1037
Type: Causal Loop
Status: Success
Time: 301ms
Energy: 2.36 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="48.42599526749788" cy="-98.53883840902509"><title>ID: PX-1038
Type: Causal Loop
Status: Kinetic Reset
Time: 268ms
Energy: 4.01 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="65.675395457216" cy="198.81534154586237"><title>ID: PX-1039
Type: Quantum Lock
Status: Success
Time: 364ms
Energy: 2.46 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="30.327136910283503" cy="-128.69867588741602"><title>ID: PX-1040
Type: Causal Loop
Status: Kinetic Reset
Time: 168ms
Energy: 4.17 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="56.873133637788804" cy="57.98538491100741"><title>ID: PX-1041
Type: Quantum Lock
Status: Kinetic Reset
Time: 315ms
Energy: 3.20 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="4.339371232855532" cy="367.17146880578235"><title>ID: PX-1042
Type: Temporal Fold
Status: Success
Time: 24ms
Energy: 1.58 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="15.544908980253604" cy="180.06258957289072"><title>ID: PX-1043
Type: Quantum Lock
Status: Success
Time: 86ms
Energy: 2.56 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="44.32290962146782" cy="242.9334486928477"><title>ID: PX-1044
Type: Temporal Fold
Status: Success
Time: 245ms
Energy: 2.23 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="38.443865112879934" cy="-214.85267543042517"><title>ID: PX-1045
Type: Entropic Drift
Status: Kinetic Reset
Time: 213ms
Energy: 4.62 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="20.2264025912372" cy="309.1277063964078"><title>ID: PX-1046
Type: Quantum Lock
Status: Success
Time: 112ms
Energy: 1.88 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="42.94194210167008" cy="-325.6812796635978"><title>ID: PX-1047
Type: Quantum Lock
Status: Kinetic Reset
Time: 238ms
Energy: 5.20 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="70.03812415444955" cy="141.25191747109776"><title>ID: PX-1048
Type: Causal Loop
Status: Success
Time: 388ms
Energy: 2.76 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="2.9644825543668185" cy="-184.240006140491"><title>ID: PX-1049
Type: Causal Loop
Status: Kinetic Reset
Time: 16ms
Energy: 4.46 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="30.964849204854254" cy="-124.27180877975651"><title>ID: PX-1050
Type: Entropic Drift
Status: Kinetic Reset
Time: 171ms
Energy: 4.15 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="54.95445064359335" cy="-92.22918654467502"><title>ID: PX-1051
Type: Quantum Lock
Status: Kinetic Reset
Time: 304ms
Energy: 3.98 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="4.818839511940605" cy="-35.96686287483578"><title>ID: PX-1052
Type: Quantum Lock
Status: Kinetic Reset
Time: 27ms
Energy: 3.69 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="40.86135394895709" cy="205.22272746529646"><title>ID: PX-1053
Type: Entropic Drift
Status: Success
Time: 226ms
Energy: 2.43 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="57.22207379313351" cy="-218.5348702636678"><title>ID: PX-1054
Type: Causal Loop
Status: Kinetic Reset
Time: 317ms
Energy: 4.64 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="12.671081928588606" cy="218.38598055821038"><title>ID: PX-1055
Type: Entropic Drift
Status: Success
Time: 70ms
Energy: 2.36 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="36.40210300325741" cy="-48.52921922677376"><title>ID: PX-1056
Type: Quantum Lock
Status: Kinetic Reset
Time: 202ms
Energy: 3.75 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="40.09946191451589" cy="352.19731935141283"><title>ID: PX-1057
Type: Temporal Fold
Status: Success
Time: 222ms
Energy: 1.66 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="35.34534842915446" cy="242.62155589166514"><title>ID: PX-1058
Type: Quantum Lock
Status: Success
Time: 196ms
Energy: 2.23 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="17.61780212056225" cy="275.16099794601763"><title>ID: PX-1059
Type: Entropic Drift
Status: Success
Time: 98ms
Energy: 2.06 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="17.79881674583805" cy="-274.50039659689685"><title>ID: PX-1060
Type: Quantum Lock
Status: Kinetic Reset
Time: 99ms
Energy: 4.93 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="18.31771647307429" cy="371.5237079212387"><title>ID: PX-1061
Type: Quantum Lock
Status: Success
Time: 101ms
Energy: 1.56 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="59.69529530594314" cy="168.21831440357607"><title>ID: PX-1062
Type: Entropic Drift
Status: Success
Time: 331ms
Energy: 2.62 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="5.39724828206797" cy="12.725324391737656"><title>ID: PX-1063
Type: Entropic Drift
Status: Kinetic Reset
Time: 30ms
Energy: 3.43 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="15.152599966037991" cy="-176.38044499683846"><title>ID: PX-1064
Type: Causal Loop
Status: Kinetic Reset
Time: 84ms
Energy: 4.42 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="6.584732473681085" cy="-244.10200408030568"><title>ID: PX-1065
Type: Causal Loop
Status: Kinetic Reset
Time: 36ms
Energy: 4.78 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="23.894172352832193" cy="-222.2807424872971"><title>ID: PX-1066
Type: Entropic Drift
Status: Kinetic Reset
Time: 132ms
Energy: 4.66 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="24.062320595307995" cy="176.64797065755735"><title>ID: PX-1067
Type: Quantum Lock
Status: Success
Time: 133ms
Energy: 2.58 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="21.484766305407977" cy="342.67866920521186"><title>ID: PX-1068
Type: Temporal Fold
Status: Success
Time: 119ms
Energy: 1.71 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="32.244740082435335" cy="-5.164860579739439"><title>ID: PX-1069
Type: Entropic Drift
Status: Kinetic Reset
Time: 179ms
Energy: 3.53 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="64.36259006208213" cy="76.17255822440914"><title>ID: PX-1070
Type: Entropic Drift
Status: Kinetic Reset
Time: 356ms
Energy: 3.10 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="-8.265799177253163" cy="83.39996737655343"><title>ID: PX-1071
Type: Quantum Lock
Status: Kinetic Reset
Time: -46ms
Energy: 3.06 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="8.232377359340543" cy="-219.87229676220787"><title>ID: PX-1072
Type: Temporal Fold
Status: Kinetic Reset
Time: 46ms
Energy: 4.65 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="45.995295113257995" cy="282.3116858633085"><title>ID: PX-1073
Type: Causal Loop
Status: Success
Time: 255ms
Energy: 2.02 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="56.605376086534655" cy="232.26904069555903"><title>ID: PX-1074
Type: Entropic Drift
Status: Success
Time: 314ms
Energy: 2.29 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="48.87267092308201" cy="-134.03021789352073"><title>ID: PX-1075
Type: Quantum Lock
Status: Kinetic Reset
Time: 271ms
Energy: 4.20 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="19.409769363642805" cy="182.73686359126418"><title>ID: PX-1076
Type: Temporal Fold
Status: Success
Time: 108ms
Energy: 2.55 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="3.9974221321996173" cy="128.5339484106744"><title>ID: PX-1077
Type: Causal Loop
Status: Kinetic Reset
Time: 22ms
Energy: 2.83 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="17.7786602583045" cy="-122.68843825224032"><title>ID: PX-1078
Type: Quantum Lock
Status: Kinetic Reset
Time: 98ms
Energy: 4.14 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="9.553295700818063" cy="230.19948342002164"><title>ID: PX-1079
Type: Causal Loop
Status: Success
Time: 53ms
Energy: 2.30 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="33.673960964694565" cy="22.73834655661246"><title>ID: PX-1080
Type: Entropic Drift
Status: Kinetic Reset
Time: 187ms
Energy: 3.38 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="4.2926329157066645" cy="281.39940000532505"><title>ID: PX-1081
Type: Causal Loop
Status: Success
Time: 24ms
Energy: 2.03 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="15.107142808059725" cy="-34.37123135685319"><title>ID: PX-1082
Type: Quantum Lock
Status: Kinetic Reset
Time: 84ms
Energy: 3.68 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="34.92285324938648" cy="169.64612216032285"><title>ID: PX-1083
Type: Quantum Lock
Status: Success
Time: 193ms
Energy: 2.61 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="22.34389790265747" cy="-128.6452593510945"><title>ID: PX-1084
Type: Temporal Fold
Status: Kinetic Reset
Time: 124ms
Energy: 4.17 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="17.691724518553585" cy="-143.90655468779218"><title>ID: PX-1085
Type: Causal Loop
Status: Kinetic Reset
Time: 98ms
Energy: 4.25 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="58.27572338575247" cy="-39.92488799097838"><title>ID: PX-1086
Type: Quantum Lock
Status: Kinetic Reset
Time: 323ms
Energy: 3.71 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="26.082329378385623" cy="383.4370852917759"><title>ID: PX-1087
Type: Quantum Lock
Status: Success
Time: 144ms
Energy: 1.50 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="23.78840930625517" cy="204.05616530724498"><title>ID: PX-1088
Type: Temporal Fold
Status: Success
Time: 132ms
Energy: 2.43 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="40.22126547983237" cy="-146.46443780770485"><title>ID: PX-1089
Type: Temporal Fold
Status: Kinetic Reset
Time: 223ms
Energy: 4.27 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="44.27555308318339" cy="-321.2814895289882"><title>ID: PX-1090
Type: Temporal Fold
Status: Kinetic Reset
Time: 245ms
Energy: 5.18 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="-4.454319996756251" cy="14.61690180373121"><title>ID: PX-1091
Type: Entropic Drift
Status: Kinetic Reset
Time: -25ms
Energy: 3.42 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="43.956666948877405" cy="-115.69290814964646"><title>ID: PX-1092
Type: Temporal Fold
Status: Kinetic Reset
Time: 243ms
Energy: 4.10 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="61.85122353324106" cy="63.29285329808886"><title>ID: PX-1093
Type: Temporal Fold
Status: Kinetic Reset
Time: 343ms
Energy: 3.17 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="36.69469949403829" cy="-263.28416392524525"><title>ID: PX-1094
Type: Quantum Lock
Status: Kinetic Reset
Time: 203ms
Energy: 4.88 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="2.197672005848859" cy="168.10573145468481"><title>ID: PX-1095
Type: Causal Loop
Status: Success
Time: 12ms
Energy: 2.62 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="32.238040693086184" cy="-107.93719948290033"><title>ID: PX-1096
Type: Entropic Drift
Status: Kinetic Reset
Time: 179ms
Energy: 4.06 J</title></circle><circle r="6" fill="rgb(245, 184, 79)" stroke="#101218" stroke-width="1" opacity="0.8" cx="12.27188063800064" cy="9.344939714609986"><title>ID: PX-1097
Type: Causal Loop
Status: Kinetic Reset
Time: 68ms
Energy: 3.45 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="10.823392795040787" cy="347.5620560424959"><title>ID: PX-1098
Type: Entropic Drift
Status: Success
Time: 60ms
Energy: 1.68 J</title></circle><circle r="6" fill="rgb(144, 229, 140)" stroke="#101218" stroke-width="1" opacity="0.8" cx="51.71182007386514" cy="271.5016056117363"><title>ID: PX-1099
Type: Quantum Lock
Status: Success
Time: 286ms
Energy: 2.08 J</title></circle></g></g></svg></div><div id="controls-root" class="widget-ui-part p-m bg-surface w-full control-grid" style="flex: 0 1 auto; --s-40: 148px; --s-30: 148px;"><div class="xxs-row standard compact" data-key="couplingK"><label class="xxs-label" title="Coupling Constant (K)" for="ctrl-gys4g" style="font-family: &quot;Google Sans&quot;, sans-serif; font-size: 14px; line-height: 20px; font-weight: 400; letter-spacing: 0px;">Coupling Constant (K)</label><div class="xxs-slider-wrap"><input type="range" class="xxs-slider" min="0.1" max="1" step="0.05" id="ctrl-gys4g" style="--progress: 100%;"></div><input type="number" class="xxs-val-pill" min="0.1" max="1" step="0.05" style="font-family: var(--ff-mono, &quot;Google Code&quot;), &quot;SF Mono&quot;, &quot;Roboto Mono&quot;, monospace; font-size: 14px; line-height: 20px; font-weight: 700; letter-spacing: 0px;"></div><div class="xxs-row compact toggle"><div class="xxs-label" title="Enable Kinetic Resets" style="font-family: &quot;Google Sans&quot;, sans-serif; font-size: 14px; line-height: 20px; font-weight: 400; letter-spacing: 0px;">Enable Kinetic Resets</div><div class="xxs-switch active" role="switch" tabindex="0" aria-checked="true"><div class="knob"></div></div></div><div class="xxs-row compact"><button class="xxs-btn" title="Randomize Runs" style="width: 100%;">Randomize Runs</button></div></div>
</body>
```
## Connections
- [[Software Engineer/Quillan-XSWE.md]]
- [[Skills/swarm-inter-agent-orchestration/swarm-inter-agent-orchestration.md]]
- [[Quillan Knowledge files/28-Multi-Agent Collective Intelligence & Social Simulation.md]]
- [[00 - Meta/03 - Training & Model.md]]
- [[Quillan Knowledge files/0-Quillan Loader Manifest.md]]
- [[system prompts/Quillan-Samurai.md]]
- [[00 - Meta/01 - Core Architecture.md]]
