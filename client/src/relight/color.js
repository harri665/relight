export const LIGHT_SWATCHES = ["#ffffff", "#ffd9a0", "#ffb26b", "#cfe0ff", "#ff5a4a", "#63e07a", "#4f8dff", "#e467ff"];
export const BRUSH_SWATCHES = ["#ffffff", "#ffd9a0", "#ff9a5a", "#9ec2ff", "#ff5a4a", "#63e07a", "#000000"];

export const hexToRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
export const rgbToHex = (c) => "#" + c.map((x) => Math.round(Math.min(1, Math.max(0, x)) * 255).toString(16).padStart(2, "0")).join("");
export const srgbToLin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
export const fmt = (x, d = 2) => Number(x).toFixed(d);
