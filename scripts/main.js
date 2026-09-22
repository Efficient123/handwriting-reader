import { system, world, Player, BlockVolume } from '@minecraft/server';
import { hidden_bias, hidden_weights, output_bias, output_weights } from './weights_and_bias';

const GRID_COUNT = 10;          
const CELL_SIZE = 10;           
const SCAN_LAYER_OFFSET = 3;    
const SCAN_TICKS_PER_ROW = 2;   
const LETTER_HOLD_TICKS = 5;    
const SCAN_BLOCK = "minecraft:yellow_concrete";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

function dotProd(vec1, vec2) {
    let total = 0;
    for (let i = 0; i < vec1.length; i++) {
        total += vec1[i] * vec2[i];
    }
    return total;
}

function sum(vec) {
    let final = 0;
    for (let i = 0; i < vec.length; i++) {
        final += vec[i];
    }
    return final;
}

function softmax(vec) {
    const final = [];
    const max = Math.max(...vec);
    const reduced = vec.map(x => Math.exp(x - max));
    const total = sum(reduced);

    for (let i = 0; i < reduced.length; i++) {
        final.push((reduced[i] / total) * 100);
    }
    return final;
}

function floorLoc(loc) {
    return {
        x: Math.floor(loc.x),
        y: Math.floor(loc.y),
        z: Math.floor(loc.z),
    };
}

function setActionBar(player, text) {
    try {
        player.onScreenDisplay.setActionBar(text);
    } catch { }
}

function clearScannerCell(dim, base, xGrid, yGrid) {
    const z = base.z + SCAN_LAYER_OFFSET;
    const x0 = base.x + xGrid * CELL_SIZE;
    const y0 = base.y - yGrid * CELL_SIZE;

    for (let y = 0; y < CELL_SIZE; y++) {
        for (let x = 0; x < CELL_SIZE; x++) {
            dim.setBlockType(
                { x: x0 + x, y: y0 - y, z },
                "minecraft:air"
            );
        }
    }
}

function clearScannerRow(dim, base, xGrid, yGrid, row) {
    const z = base.z + SCAN_LAYER_OFFSET;
    const x0 = base.x + xGrid * CELL_SIZE;
    const y0 = base.y - yGrid * CELL_SIZE;

    for (let x = 0; x < CELL_SIZE; x++) {
        dim.setBlockType(
            { x: x0 + x, y: y0 - row, z },
            "minecraft:air"
        );
    }
}

function drawScannerRow(dim, base, xGrid, yGrid, row) {
    const z = base.z + SCAN_LAYER_OFFSET;
    const x0 = base.x + xGrid * CELL_SIZE;
    const y0 = base.y - yGrid * CELL_SIZE;

    for (let x = 0; x < CELL_SIZE; x++) {
        dim.setBlockType(
            { x: x0 + x, y: y0 - row, z },
            SCAN_BLOCK
        );
    }
}

function readGlyphVector(dim, topLeft, xGrid, yGrid) {
    const vector100 = [];

    for (let y = 0; y < CELL_SIZE; y++) {
        for (let x = 0; x < CELL_SIZE; x++) {
            const block = dim.getBlock({
                x: topLeft.x + x + (CELL_SIZE * xGrid),
                y: topLeft.y - y - (CELL_SIZE * yGrid),
                z: topLeft.z + 1,
            });

            if (!block || block.typeId !== "minecraft:white_concrete") {
                vector100.push(0);
            } else {
                vector100.push(1);
            }
        }
    }

    return vector100;
}

function predictLetter(vector100) {
    const hidden_output = [];

    for (let i = 0; i < 48; i++) {
        hidden_output.push(
            Math.max(0, dotProd(vector100, hidden_weights[i]) + hidden_bias[i])
        );
    }

    const output = [];
    for (let i = 0; i < 26; i++) {
        output.push(dotProd(hidden_output, output_weights[i]) + output_bias[i]);
    }

    const prob = softmax(output);

    let bestIndex = 0;
    for (let i = 1; i < prob.length; i++) {
        if (prob[i] > prob[bestIndex]) bestIndex = i;
    }

    return {
        letter: LETTERS[bestIndex],
        probability: prob[bestIndex],
        all: prob
    };
}

function buildSheet(dim, topLeft) {
    const volume = new BlockVolume(
        topLeft,
        { x: topLeft.x + (GRID_COUNT * CELL_SIZE), y: topLeft.y - (GRID_COUNT * CELL_SIZE), z: topLeft.z }
    );

    dim.fillBlocks(volume, "minecraft:black_concrete");

    const colors = ["red", "blue", "lime", "yellow"];

    for (let yGrid = 0; yGrid < GRID_COUNT; yGrid++) {
        for (let xGrid = 0; xGrid < GRID_COUNT; xGrid++) {
            const color = colors[(xGrid + yGrid) % colors.length];

            for (let y = 0; y < CELL_SIZE; y++) {
                for (let x = 0; x < CELL_SIZE; x++) {
                    if (x !== 0 && x !== CELL_SIZE - 1 && y !== 0 && y !== CELL_SIZE - 1) continue;

                    dim.setBlockType(
                        {
                            x: topLeft.x + x + xGrid * CELL_SIZE,
                            y: topLeft.y - y - yGrid * CELL_SIZE,
                            z: topLeft.z + 2
                        },
                        `minecraft:${color}_stained_glass`
                    );
                }
            }
        }
    }
}

const playerList = new Map();

let topLeft = null;
let bottomRight = null;
let dimIn = null;

let readerSession = {
    active: false,
    player: null,
    dim: null,
    cells: [],
    cellIndex: 0,
    scanRow: 0,
    cooldown: 0,
    phase: "idle", 
    output: "",
};

function startReaderSession(player) {
    if (!topLeft || !bottomRight) return;
    if (readerSession.active) return;

    const dim = player.dimension;

    readerSession = {
        active: true,
        player,
        dim,
        cells: [],
        cellIndex: 0,
        scanRow: 0,
        cooldown: 0,
        phase: "scan",
        output: "",
    };

    for (let yGrid = 0; yGrid < GRID_COUNT; yGrid++) {
        for (let xGrid = 0; xGrid < GRID_COUNT; xGrid++) {
            readerSession.cells.push({ xGrid, yGrid });
        }
    }

    setActionBar(player, "§eReading...");
}

world.afterEvents.itemStartUse.subscribe((event) => {
    if (!event.itemStack.typeId.includes('spear')) return;
    if (!(event.source instanceof Player)) return;
    playerList.set(event.source.name, event.source);
});

world.afterEvents.itemStopUse.subscribe((event) => {
    if (!event.itemStack.typeId.includes('spear')) return;
    if (!(event.source instanceof Player)) return;
    playerList.delete(event.source.name);
});

world.afterEvents.chatSend.subscribe((event) => {
    const msg = event.message;
    const ply = event.sender;

    if (msg === "!sheet") {
        const loc = floorLoc(ply.location);
        const dim = ply.dimension;

        topLeft = {
            x: loc.x,
            y: loc.y + 100,
            z: loc.z
        };

        bottomRight = {
            x: loc.x + 100,
            y: loc.y,
            z: loc.z
        };

        dimIn = dim;
        buildSheet(dim, topLeft);
        setActionBar(ply, "§aSheet built.");
    }

    if (msg === "!guess") {
        startReaderSession(ply);
    }
});

system.runInterval(() => {
    for (const p of [...playerList.keys()]) {
        const player = playerList.get(p);
        if (!player) continue;

        const blockRaycast = player.getBlockFromViewDirection({ maxDistance: 100 });
        if (!blockRaycast) continue;
        if (blockRaycast.block.typeId !== "minecraft:black_concrete") continue;

        const dim = player.dimension;
        const blockLoc = blockRaycast.block.location;

        dim.setBlockType(
            { x: blockLoc.x, y: blockLoc.y, z: blockLoc.z + 1 },
            "minecraft:white_concrete"
        );
    }

    if (!readerSession.active) return;

    if (readerSession.cooldown > 0) {
        readerSession.cooldown--;
        return;
    }

    const job = readerSession.cells[readerSession.cellIndex];
    if (!job) {
        if (readerSession.player) {
            setActionBar(readerSession.player, "§aDone.");
        }

        world.sendMessage(readerSession.output.trim());

        if (topLeft && bottomRight && dimIn) {
            try {
                dimIn.fillBlocks(new BlockVolume(topLeft, bottomRight), "minecraft:air");
            } catch { }
        }

        readerSession.active = false;
        readerSession.phase = "idle";
        return;
    }

    const { xGrid, yGrid } = job;

    if (readerSession.phase === "scan") {
        if (readerSession.scanRow === 0) {
            clearScannerCell(readerSession.dim, topLeft, xGrid, yGrid);
        }

        if (readerSession.scanRow > 0) {
            clearScannerRow(readerSession.dim, topLeft, xGrid, yGrid, readerSession.scanRow - 1);
        }

        drawScannerRow(readerSession.dim, topLeft, xGrid, yGrid, readerSession.scanRow);

        const displayText = readerSession.output + "▌";
        if (readerSession.player) {
            setActionBar(readerSession.player, `§eReading: §f${displayText}`);
        }

        readerSession.scanRow++;
        readerSession.cooldown = SCAN_TICKS_PER_ROW - 1;

        if (readerSession.scanRow >= CELL_SIZE) {
            readerSession.phase = "predict";
            readerSession.cooldown = 1;
        }

        return;
    }

    if (readerSession.phase === "predict") {
        const vector100 = readGlyphVector(readerSession.dim, topLeft, xGrid, yGrid);

        if (vector100.indexOf(1) === -1) {
            readerSession.output += " ";
            if (readerSession.player) {
                setActionBar(readerSession.player, `§eReading: §f${readerSession.output}▌`);
            }
        } else {
            const predicted = predictLetter(vector100);
            readerSession.output += predicted.letter;

            if (readerSession.player) {
                setActionBar(
                    readerSession.player,
                    `§eReading: §f${readerSession.output} §7(${predicted.probability.toFixed(1)}%)`
                );
            }
        }

        clearScannerCell(readerSession.dim, topLeft, xGrid, yGrid);

        readerSession.cellIndex++;
        readerSession.scanRow = 0;
        readerSession.phase = "scan";
        readerSession.cooldown = LETTER_HOLD_TICKS;
        return;
    }
}, 1);

system.beforeEvents.shutdown.subscribe(() => {
    system.run(() => {
        if (dimIn && topLeft && bottomRight) {
            dimIn.fillBlocks(new BlockVolume(topLeft, bottomRight), "minecraft:air");
        }
    });
});