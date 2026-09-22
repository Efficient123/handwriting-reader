import { system, world, Player, BlockVolume } from '@minecraft/server';
import { hidden_bias, hidden_weights, output_bias, output_weights } from './weights_and_bias';

const GRID_COUNT = 10;
const CELL_SIZE = 10;
const SHEET_SIZE = GRID_COUNT * CELL_SIZE;

const SCAN_LAYER_OFFSET = 3;
const SCAN_TICKS_PER_ROW = 2;
const LETTER_HOLD_TICKS = 5;
const SCAN_BLOCK = "minecraft:yellow_concrete";

const SHEET_PROPERTY = "letter_reader:sheet";

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



function savePlayerSheet(player, sheet) {
    try {
        player.setDynamicProperty(
            SHEET_PROPERTY,
            JSON.stringify(sheet)
        );

        return true;
    } catch (error) {
        console.warn(`Failed to save sheet for ${player.name}: ${error}`);
        return false;
    }
}

function loadPlayerSheet(player) {
    try {
        const raw = player.getDynamicProperty(SHEET_PROPERTY);

        if (typeof raw !== "string") {
            return null;
        }

        const sheet = JSON.parse(raw);

        if (
            typeof sheet?.x !== "number" ||
            typeof sheet?.y !== "number" ||
            typeof sheet?.z !== "number" ||
            typeof sheet?.dimension !== "string"
        ) {
            return null;
        }

        return sheet;
    } catch (error) {
        console.warn(`Failed to load sheet for ${player.name}: ${error}`);
        return null;
    }
}

function deletePlayerSheet(player) {
    try {
        player.setDynamicProperty(SHEET_PROPERTY, undefined);
        return true;
    } catch (error) {
        console.warn(`Failed to delete sheet for ${player.name}: ${error}`);
        return false;
    }
}

function getDimension(id) {
    try {
        return world.getDimension(id);
    } catch {
        return null;
    }
}


function getSheetInfo(player) {
    const saved = loadPlayerSheet(player);

    if (!saved) {
        return null;
    }

    const dim = getDimension(saved.dimension);

    if (!dim) {
        return null;
    }

    return {
        topLeft: {
            x: saved.x,
            y: saved.y,
            z: saved.z,
        },

        bottomRight: {
            x: saved.x + SHEET_SIZE,
            y: saved.y - SHEET_SIZE,
            z: saved.z,
        },

        dim,
        dimension: saved.dimension,
    };
}



const allocatedPlots = new Set();

function getPlotKey(dimension, x, y, z) {
    return `${dimension}:${x}:${y}:${z}`;
}

function reserveExistingSheets() {
    allocatedPlots.clear();

    for (const player of world.getPlayers()) {
        const sheet = loadPlayerSheet(player);

        if (!sheet) continue;

        allocatedPlots.add(
            getPlotKey(
                sheet.dimension,
                sheet.x,
                sheet.y,
                sheet.z
            )
        );
    }
}

function findNewPlot(player) {
    reserveExistingSheets();

    const loc = floorLoc(player.location);
    const dimension = player.dimension.id;

    
    const spacing = SHEET_SIZE + 10;

    for (let i = 0; i < 10000; i++) {
        const x = loc.x + i * spacing;
        const y = loc.y + 100;
        const z = loc.z;

        const key = getPlotKey(
            dimension,
            x,
            y,
            z
        );

        if (!allocatedPlots.has(key)) {
            return {
                x,
                y,
                z,
                dimension,
            };
        }
    }

    return null;
}



function clearScannerCell(dim, base, xGrid, yGrid) {
    const z = base.z + SCAN_LAYER_OFFSET;
    const x0 = base.x + xGrid * CELL_SIZE;
    const y0 = base.y - yGrid * CELL_SIZE;

    for (let y = 0; y < CELL_SIZE; y++) {
        for (let x = 0; x < CELL_SIZE; x++) {
            dim.setBlockType(
                {
                    x: x0 + x,
                    y: y0 - y,
                    z
                },
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
            {
                x: x0 + x,
                y: y0 - row,
                z
            },
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
            {
                x: x0 + x,
                y: y0 - row,
                z
            },
            SCAN_BLOCK
        );
    }
}

function readGlyphVector(dim, topLeft, xGrid, yGrid) {
    const vector100 = [];

    for (let y = 0; y < CELL_SIZE; y++) {
        for (let x = 0; x < CELL_SIZE; x++) {
            const block = dim.getBlock({
                x: topLeft.x + x + CELL_SIZE * xGrid,
                y: topLeft.y - y - CELL_SIZE * yGrid,
                z: topLeft.z + 1,
            });

            if (
                !block ||
                block.typeId !== "minecraft:white_concrete"
            ) {
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
            Math.max(
                0,
                dotProd(vector100, hidden_weights[i]) +
                hidden_bias[i]
            )
        );
    }

    const output = [];

    for (let i = 0; i < 26; i++) {
        output.push(
            dotProd(hidden_output, output_weights[i]) +
            output_bias[i]
        );
    }

    const prob = softmax(output);

    let bestIndex = 0;

    for (let i = 1; i < prob.length; i++) {
        if (prob[i] > prob[bestIndex]) {
            bestIndex = i;
        }
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
        {
            x: topLeft.x + SHEET_SIZE,
            y: topLeft.y - SHEET_SIZE,
            z: topLeft.z
        }
    );

    dim.fillBlocks(
        volume,
        "minecraft:black_concrete"
    );

    const colors = [
        "red",
        "blue",
        "lime",
        "yellow"
    ];

    for (let yGrid = 0; yGrid < GRID_COUNT; yGrid++) {
        for (let xGrid = 0; xGrid < GRID_COUNT; xGrid++) {
            const color =
                colors[(xGrid + yGrid) % colors.length];

            for (let y = 0; y < CELL_SIZE; y++) {
                for (let x = 0; x < CELL_SIZE; x++) {
                    if (
                        x !== 0 &&
                        x !== CELL_SIZE - 1 &&
                        y !== 0 &&
                        y !== CELL_SIZE - 1
                    ) {
                        continue;
                    }

                    dim.setBlockType(
                        {
                            x:
                                topLeft.x +
                                x +
                                xGrid * CELL_SIZE,

                            y:
                                topLeft.y -
                                y -
                                yGrid * CELL_SIZE,

                            z: topLeft.z + 2
                        },

                        `minecraft:${color}_stained_glass`
                    );
                }
            }
        }
    }
}



function createSheetForPlayer(player) {
    const existing = getSheetInfo(player);

    if (existing) {
        return {
            created: false,
            sheet: existing
        };
    }

    const saved = findNewPlot(player);

    if (!saved) {
        setActionBar(
            player,
            "§cNo available sheet locations."
        );

        return null;
    }

    const dim = getDimension(saved.dimension);

    if (!dim) {
        setActionBar(
            player,
            "§cCould not find the sheet dimension."
        );

        return null;
    }

    const topLeft = {
        x: saved.x,
        y: saved.y,
        z: saved.z
    };

    buildSheet(dim, topLeft);

    if (!savePlayerSheet(player, saved)) {
        setActionBar(
            player,
            "§cSheet created, but failed to save its location."
        );

        return null;
    }

    return {
        created: true,

        sheet: {
            topLeft,

            bottomRight: {
                x: saved.x + SHEET_SIZE,
                y: saved.y - SHEET_SIZE,
                z: saved.z
            },

            dim,
            dimension: saved.dimension,
        }
    };
}

function clearPlayerSheet(player) {
    const sheet = getSheetInfo(player);

    if (!sheet) {
        setActionBar(
            player,
            "§7You don't have a sheet."
        );

        return false;
    }

    try {
        sheet.dim.fillBlocks(
            new BlockVolume(
                { x: sheet.topLeft.x, y: sheet.topLeft.y, z: sheet.topLeft.z, },
                { x: sheet.bottomRight.x, y: sheet.bottomRight.y, z: sheet.bottomRight.z + 2, },
            ),
            "minecraft:air"
        );
    } catch (error) {
        console.warn(
            `Failed to clear sheet for ${player.name}: ${error}`
        );

        setActionBar(
            player,
            "§cFailed to clear your sheet."
        );

        return false;
    }

    deletePlayerSheet(player);

    setActionBar(
        player,
        "§aSheet cleared."
    );

    return true;
}



let readerSession = {
    active: false,
    player: null,
    dim: null,
    topLeft: null,
    bottomRight: null,

    cells: [],
    cellIndex: 0,
    scanRow: 0,
    cooldown: 0,

    phase: "idle",
    output: "",
};

function startReaderSession(player) {
    if (readerSession.active) {
        setActionBar(
            player,
            "§cSomeone is already using the reader."
        );

        return;
    }

    const sheet = getSheetInfo(player);

    if (!sheet) {
        setActionBar(
            player,
            "§cYou don't have a sheet. Use !sheet first."
        );

        return;
    }

    readerSession = {
        active: true,

        player,
        dim: sheet.dim,
        topLeft: sheet.topLeft,
        bottomRight: sheet.bottomRight,

        cells: [],
        cellIndex: 0,
        scanRow: 0,
        cooldown: 0,

        phase: "scan",
        output: "",
    };

    for (let yGrid = 0; yGrid < GRID_COUNT; yGrid++) {
        for (let xGrid = 0; xGrid < GRID_COUNT; xGrid++) {
            readerSession.cells.push({
                xGrid,
                yGrid
            });
        }
    }

    setActionBar(
        player,
        "§eReading..."
    );
}



const playerList = new Map();

world.afterEvents.itemStartUse.subscribe((event) => {
    if (!event.itemStack.typeId.includes('spear')) return;
    if (!(event.source instanceof Player)) return;

    playerList.set(
        event.source.id,
        event.source
    );
});

world.afterEvents.itemStopUse.subscribe((event) => {
    if (!event.itemStack.typeId.includes('spear')) return;
    if (!(event.source instanceof Player)) return;

    playerList.delete(
        event.source.id
    );
});



world.afterEvents.chatSend.subscribe((event) => {
    const msg = event.message.trim();
    const player = event.sender;

    
    if (msg === "!sheet") {
        const existing = getSheetInfo(player);

        if (existing) {
            setActionBar(
                player,
                "§aYour sheet is already saved."
            );

            return;
        }

        const result = createSheetForPlayer(player);

        if (!result) return;

        if (result.created) {
            setActionBar(
                player,
                "§aSheet created and saved."
            );
        }

        return;
    }

    
    if (msg === "!guess") {
        startReaderSession(player);
        return;
    }

    
    if (msg === "!clear") {
        
        if (
            readerSession.active &&
            readerSession.player?.id === player.id
        ) {
            setActionBar(
                player,
                "§cYou can't clear the sheet while reading."
            );

            return;
        }

        clearPlayerSheet(player);
        return;
    }
});



system.runInterval(() => {

    
    for (const [playerId, player] of playerList) {
        if (!player) {
            playerList.delete(playerId);
            continue;
        }

        const blockRaycast =
            player.getBlockFromViewDirection({
                maxDistance: 100
            });

        if (!blockRaycast) continue;

        if (
            blockRaycast.block.typeId !==
            "minecraft:black_concrete"
        ) {
            continue;
        }

        const dim = player.dimension;
        const blockLoc = blockRaycast.block.location;

        dim.setBlockType(
            {
                x: blockLoc.x,
                y: blockLoc.y,
                z: blockLoc.z + 1
            },

            "minecraft:white_concrete"
        );
    }

    
    if (!readerSession.active) return;

    
    if (readerSession.cooldown > 0) {
        readerSession.cooldown--;
        return;
    }

    const job =
        readerSession.cells[
        readerSession.cellIndex
        ];

    
    if (!job) {
        if (readerSession.player) {
            setActionBar(
                readerSession.player,
                "§aDone."
            );
        }

        world.sendMessage(
            readerSession.output.trim()
        );

        
        if (
            readerSession.dim &&
            readerSession.topLeft
        ) {
            for (
                let yGrid = 0;
                yGrid < GRID_COUNT;
                yGrid++
            ) {
                for (
                    let xGrid = 0;
                    xGrid < GRID_COUNT;
                    xGrid++
                ) {
                    clearScannerCell(
                        readerSession.dim,
                        readerSession.topLeft,
                        xGrid,
                        yGrid
                    );
                }
            }
        }

        readerSession.active = false;
        readerSession.phase = "idle";

        return;
    }

    const {
        xGrid,
        yGrid
    } = job;

    

    if (readerSession.phase === "scan") {

        if (readerSession.scanRow === 0) {
            clearScannerCell(
                readerSession.dim,
                readerSession.topLeft,
                xGrid,
                yGrid
            );
        }

        if (readerSession.scanRow > 0) {
            clearScannerRow(
                readerSession.dim,
                readerSession.topLeft,
                xGrid,
                yGrid,
                readerSession.scanRow - 1
            );
        }

        drawScannerRow(
            readerSession.dim,
            readerSession.topLeft,
            xGrid,
            yGrid,
            readerSession.scanRow
        );

        const displayText =
            readerSession.output + "▌";

        if (readerSession.player) {
            setActionBar(
                readerSession.player,
                `§eReading: §f${displayText}`
            );
        }

        readerSession.scanRow++;

        readerSession.cooldown =
            SCAN_TICKS_PER_ROW - 1;

        if (
            readerSession.scanRow >= CELL_SIZE
        ) {
            readerSession.phase = "predict";
            readerSession.cooldown = 1;
        }

        return;
    }

    

    if (readerSession.phase === "predict") {

        const vector100 = readGlyphVector(
            readerSession.dim,
            readerSession.topLeft,
            xGrid,
            yGrid
        );

        if (vector100.indexOf(1) === -1) {

            readerSession.output += " ";

            if (readerSession.player) {
                setActionBar(
                    readerSession.player,
                    `§eReading: §f${readerSession.output}▌`
                );
            }

        } else {

            const predicted =
                predictLetter(vector100);

            readerSession.output +=
                predicted.letter;

            if (readerSession.player) {
                setActionBar(
                    readerSession.player,

                    `§eReading: §f${readerSession.output} ` +
                    `§7(${predicted.probability.toFixed(1)}%)`
                );
            }
        }

        
        clearScannerCell(
            readerSession.dim,
            readerSession.topLeft,
            xGrid,
            yGrid
        );

        readerSession.cellIndex++;
        readerSession.scanRow = 0;
        readerSession.phase = "scan";
        readerSession.cooldown =
            LETTER_HOLD_TICKS;

        return;
    }

}, 1);
