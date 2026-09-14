(() => {
    'use strict';

    const videoSource = document.getElementById('videoSource');
    const streamImg = document.getElementById('streamImg');
    const countValue = document.getElementById('countValue');
    const statusBadge = document.getElementById('statusBadge');
    const fpsValue = document.getElementById('fpsValue');
    const latencyValue = document.getElementById('latencyValue');

    // Chart.js Configuration
    const MAX_DATA_POINTS = 30;
    const initialLabels = Array(MAX_DATA_POINTS).fill('');
    const initialData = Array(MAX_DATA_POINTS).fill(0);

    const ctx = document.getElementById('crowdChart').getContext('2d');
    const crowdChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: initialLabels,
            datasets: [{
                label: 'Detected Head Count',
                data: initialData,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.08)',
                borderWidth: 2,
                fill: true,
                tension: 0.25,
                pointRadius: 2,
                pointBackgroundColor: '#2563eb'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMax: 20,
                    grid: { color: '#f1f5f9' },
                    ticks: { precision: 0, font: { size: 11 } }
                },
                x: {
                    grid: { display: false },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    let lastChartUpdate = Date.now();

    // Poll live metrics from Flask every 500ms
    async function updateMetrics() {
        try {
            const res = await fetch('/api/metrics', { cache: 'no-store' });
            if (!res.ok) return;
            const data = await res.json();

            const count = data.count ?? 0;
            const status = data.status || 'NORMAL';
            const fps = Number(data.fps ?? 0).toFixed(1);
            const latency = Number(data.inference_time_ms ?? 0).toFixed(1);

            // Update numerical values
            countValue.innerText = count;
            fpsValue.innerText = fps;
            latencyValue.innerText = latency;

            // Update status badge
            statusBadge.innerText = status;
            statusBadge.className = 'density-badge ' + (
                status === 'DANGER' ? 'badge-danger' : 
                (status === 'WARNING' ? 'badge-warning' : 'badge-normal')
            );

            // Update chart once every second
            const now = Date.now();
            if (now - lastChartUpdate >= 1000) {
                crowdChart.data.labels.push('');
                crowdChart.data.labels.shift();

                crowdChart.data.datasets[0].data.push(count);
                crowdChart.data.datasets[0].data.shift();

                // Dynamic chart color based on density severity
                if (status === 'DANGER') {
                    crowdChart.data.datasets[0].borderColor = '#dc2626';
                    crowdChart.data.datasets[0].backgroundColor = 'rgba(220, 38, 38, 0.08)';
                    crowdChart.data.datasets[0].pointBackgroundColor = '#dc2626';
                } else if (status === 'WARNING') {
                    crowdChart.data.datasets[0].borderColor = '#d97706';
                    crowdChart.data.datasets[0].backgroundColor = 'rgba(217, 119, 6, 0.08)';
                    crowdChart.data.datasets[0].pointBackgroundColor = '#d97706';
                } else {
                    crowdChart.data.datasets[0].borderColor = '#16a34a';
                    crowdChart.data.datasets[0].backgroundColor = 'rgba(22, 163, 74, 0.08)';
                    crowdChart.data.datasets[0].pointBackgroundColor = '#16a34a';
                }

                crowdChart.update();
                lastChartUpdate = now;
            }

        } catch (err) {
            console.error("Telemetry error:", err);
        }
    }

    setInterval(updateMetrics, 500);
    updateMetrics();

    // Handle dropdown video switching
    videoSource.addEventListener('change', async (e) => {
        const source = e.target.value;
        videoSource.disabled = true;

        try {
            await fetch('/api/set_source', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source })
            });

            // Reconnect stream
            const base = streamImg.src.split('?')[0];
            streamImg.src = `${base}?t=${Date.now()}`;

            // Reset chart dataset for clean video transition
            crowdChart.data.datasets[0].data = Array(MAX_DATA_POINTS).fill(0);
            crowdChart.update();
        } catch (err) {
            console.error("Failed to switch video:", err);
        } finally {
            videoSource.disabled = false;
        }
    });
})();

// Reliable CSV Blob Exporter
const exportBtn = document.getElementById('exportBtn');
if (exportBtn) {
    exportBtn.addEventListener('click', async () => {
        exportBtn.innerText = 'Exporting...';
        exportBtn.disabled = true;

        try {
            const response = await fetch('/api/export_logs', { cache: 'no-store' });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = downloadUrl;
            a.download = `crowd_telemetry_logs_${Date.now()}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(downloadUrl);
            a.remove();
        } catch (err) {
            console.error('Export failed:', err);
            alert('Failed to download logs. Check server terminal.');
        } finally {
            exportBtn.innerText = 'Export CSV';
            exportBtn.disabled = false;
        }
    });
}