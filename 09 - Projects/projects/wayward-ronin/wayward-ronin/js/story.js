// story.js — Kirisato's story: intro, three decision events, reputation,
// and six endings. ALL writing is original. Mechanics follow the
// open-ended-samurai formula (town hub, factions, branching choices,
// multiple endings) without using any copyrighted names, characters, or text.

export const ENDINGS = {
  clan: {
    title: 'THE MAGISTRATE\u2019S BLADE',
    epilogue: `Ren's headband hangs from the manor gate by morning.\n\nMagistrate Gendo keeps his word. The taxes continue, but the storehouses are safe, the roads are quiet, and your purse is heavy with clan silver.\n\nIn the market, mothers pull their children inside when you pass. Order has a price, and Kirisato pays it — to you, now.\n\nYou have a master again. You tell yourself that was the point.`,
  },
  rebel: {
    title: 'ASHEN DAWN',
    epilogue: `The storehouse burns until dawn, and the clan's wagons roll out of Kirisato by week's end.\n\nRen clasps your shoulder and calls you brother. The town eats its own rice for the first time in years.\n\nBut freedom, it turns out, also needs feeding — and the Ashen Blades are already arguing about who decides the portions.\n\nYou won the town its dawn. What it does with the daylight is no longer yours to decide.`,
  },
  town: {
    title: 'THE LANTERN KEEPER',
    epilogue: `You never drew your sword for either side — you drew it for the people between them.\n\nWhen the clan came to collect, you stood in the road. When the rebels came to burn, you stood in the road. Eventually both sides learned: the road belongs to Kirisato.\n\nMei's stall prospers. Elder Jiro lights an extra lantern each dusk — "for the stray," he says. Kiku practices with a wooden sword you carved her, and tells everyone a real samurai taught her.\n\nYou came to Kirisato with nothing. You leave it with a town that remembers your name.`,
  },
  wolf: {
    title: 'THE ROAD GOES ON',
    epilogue: `You walk north out of Kirisato as the dusk deepens, and you do not look back.\n\nBehind you, the clan still taxes and the rebels still burn, and the townsfolk still endure — none of it yours to fix. A rōnin's road is not a home; it is the space between homes.\n\nSomewhere ahead there is another town, another dusk, another choice.\n\nThe road does not judge. That is why you walk it.`,
  },
  butcher: {
    title: 'THE BUTCHER OF KIRISATO',
    epilogue: `Clan red and rebel ash — both stain your blade now, and neither side claims you.\n\nKirisato locks its doors at dusk. Not because of the magistrate. Not because of the rebels.\n\nBecause of you.\n\nMothers tell their children: if you wander after dark, the Butcher will find you. Kiku doesn't practice with her stick anymore.\n\nYou wanted to be free of masters. Congratulations. You are free of everything.`,
  },
  death: {
    title: 'CUT DOWN',
    epilogue: `The dusk takes you the way it takes all stray blades — quietly, in a town that was never yours.\n\nElder Jiro pays for the burial. Mei leaves a rice ball on the grave. Nobody knows your name, so the marker reads only: A RŌNIN.\n\nThe road goes on without you. It always does.`,
  },
};

export class Story {
  constructor(ctx) {
    // ctx: { rep, flags, audio, fx, dialogue, npcs, spawnEnemies, setObjective,
    //        banner, toast, endGame, heal, player, hideNpc, showWorldActors }
    this.ctx = ctx;
    this.rep = { clan: 0, rebel: 0, town: 0 };
    this.flags = {};
    this.kills = { clan: 0, rebel: 0 };
    this.ended = false;
    this.trees = this.buildTrees();
  }

  adjust(faction, amt) {
    const c = this.ctx;
    this.rep[faction] = Math.max(-100, Math.min(100, this.rep[faction] + amt));
    if (amt > 0) { c.toast(`${label(faction)} +${amt}`); c.audio.repUp(); }
    else if (amt < 0) { c.toast(`${label(faction)} ${amt}`); c.audio.repDown(); }
    c.updateRepHud(this.rep);
  }

  // ---------- dialogue trees ----------
  buildTrees() {
    const S = this; // for closures
    return {
      jiro_intro: {
        speaker: 'Elder Jiro', plate: '#e8c96a', start: 's',
        nodes: {
          s: { text: `So. Another stray blade drifts in with the evening wind.\n\nTell me, rōnin — what does a swordsman want in Kirisato? The war's been over for years. All that's left here is taxes and ghosts.`,
            choices: [
              { text: `Just passing through.`, to: 'n1' },
              { text: `I'm looking for work.`, to: 'n1b' },
              { text: `Trouble. I can smell it.`, to: 'n1c' },
            ] },
          n1: { text: `Everyone says that. This town has a way of keeping people — usually in its graveyard.`, to: 'n2' },
          n1b: { text: `Work? The magistrate always needs blades. The rebels need them too, though they'd cut their own tongues out before saying it aloud.`, to: 'n2' },
          n1c: { text: `Then your nose works better than the magistrate's spies. The clan squeezes, the rebels burn, and we're the grain between the millstones.`, to: 'n2' },
          n2: { text: `Listen well. Magistrate Gendo's clan bleeds this town with "protection" taxes. The Ashen Blades — rebels, rōnin like you once were — raid the clan's storehouses and call it justice.\n\nThe rest of us just try to eat. You'll be asked to choose, sooner than you think.\n\nWhatever you do... don't choose lightly.`,
            onEnter: (c) => {
              S.flags.introDone = true;
              c.setObjective('Learn the town — talk to its people, then follow the trouble');
              c.banner('KIRISATO');
            } },
        },
      },
      jiro_idle: {
        speaker: 'Elder Jiro', plate: '#e8c96a', start: 's',
        nodes: {
          s: { text: (c) => S.jiroHint(),
            choices: [{ text: `I'll keep that in mind.`, to: null }] },
        },
      },
      mei: {
        speaker: 'Mei the Merchant', plate: '#e8c96a', start: 's',
        nodes: {
          s: { text: `A blade! Don't mind the mess — ...you're not with the tax collectors, are you? No? Then buy something or keep walking, friend.`,
            choices: [
              { text: `A rice ball, please. (restores vitality)`, to: 'buy', if: () => !S.flags.boughtFood,
                do: (c) => { S.flags.boughtFood = true; c.heal(35); c.toast('Ate a warm rice ball. +35 vitality'); } },
              { text: `What do the collectors do here?`, to: 'tax' },
              { text: `Just browsing.`, to: null },
            ] },
          buy: { text: `Fresh this morning. Eat, eat — a swordsman with an empty stomach makes poor decisions.`, choices: [{ text: `Thanks, Mei.`, to: null }] },
          tax: { text: `What don't they do? "Protection" tax, road tax, lantern tax — last month they taxed my shadow. Captain Isamu's men took half my stock when I couldn't pay.\n\nIf you're the intervening type... they come around the market most evenings.`, choices: [{ text: `I'll remember that.`, to: null }] },
        },
      },
      kiku: {
        speaker: 'Kiku', plate: '#e8c96a', start: 's',
        nodes: {
          s: { text: `Are you a samurai?! You have a REAL sword! When I grow up I'm gonna have a sword and nobody will take our rice ever again!`,
            choices: [
              { text: `"Keep practicing, little warrior."`, to: 'kind', do: () => S.adjust('town', 5) },
              { text: `"War isn't a game, kid."`, to: 'stern' },
              { text: `"Maybe. If you're lucky, you'll never need one."`, to: null },
            ] },
          kind: { text: `Hee! Did you hear that?! A real samurai said I'm a warrior!`, choices: [{ text: `(smile)`, to: null }] },
          stern: { text: `...Oh. ...Okay.`, choices: [{ text: `(walk away)`, to: null }] },
        },
      },
      isamu: {
        speaker: 'Captain Isamu', plate: '#e08080', start: 's',
        nodes: {
          s: { text: (c) => S.flags.reported
              ? `Back again? The magistrate appreciates... cooperative blades. Keep your eyes open for rebel filth.`
              : `State your business, rōnin. The magistrate's patience for wandering blades is thin.`,
            choices: [
              { text: `Just looking around.`, to: null },
              { text: `The town seems tense.`, to: 'tense', if: () => !S.flags.reported },
              { text: `The rebels — where do they hide?`, to: 'ask', if: () => !S.flags.reported },
            ] },
          tense: { text: `Tense? The Ashen Blades burn our storehouses and you call it tension.\n\nIf you've seen their kind — skulking, whispering — Captain Isamu pays for information. Remember that.`, choices: [{ text: `I'll remember.`, to: null }] },
          ask: { text: `Bold question for a stray. West alley, if the rumors are worth anything. But walk in there waving a clan banner and you'll come out in pieces.\n\n...Unless you bring me something useful first.`, choices: [{ text: `We'll see.`, to: null }] },
        },
      },
      ren: {
        speaker: 'Ren of the Ashen Blades', plate: '#9dc08a', start: 's',
        nodes: {
          s: { text: (c) => {
              if (S.flags.event2done === 'report') return `So. You're the magistrate's dog now. Tsubaki told me. Get out of my sight before I forget we were ever the same kind.`;
              if (!S.flags.event1done) return `Another rōnin. This town's crawling with us lately — like crows after a battle.\n\nWhat do you want?`;
              if (!S.flags.event2done) return `You've got the look of someone who's seen the market's little tax ceremony.\n\nThe clan's rice storehouse — east side, fat with stolen harvest. Tonight, we take it back. I need a blade I can trust at my back.\n\nWill you burn it with me?`;
              return `The storehouse job is done, one way or another. Whatever comes next — walk carefully, brother.`;
            },
            choices: [
              { text: `Who are the Ashen Blades?`, to: 'who', if: () => !S.flags.event1done },
              { text: `I'll burn it with you.`, to: 'accept', if: () => S.flags.event1done && !S.flags.event2done,
                do: (c) => S.event2Choice('accept', c) },
              { text: `No. I won't steal from anyone.`, to: 'refuse', if: () => S.flags.event1done && !S.flags.event2done,
                do: (c) => S.event2Choice('refuse', c) },
              { text: `I'll tell Captain Isamu about this.`, to: 'report', if: () => S.flags.event1done && !S.flags.event2done,
                do: (c) => S.event2Choice('report', c) },
              { text: `Just passing through.`, to: null, if: () => !S.flags.event1done || S.flags.event2done },
            ] },
          who: { text: `We're what's left when the clan takes everything. They tax the rice, we take it back. The magistrate calls it treason.\n\nWe call it dinner.`, choices: [{ text: `I see.`, to: null }] },
          accept: { text: `Good. The storehouse, east side — go now, before they move the grain. If guards come, they come. That's what the sword is for.\n\nAnd rōnin... thank you.`, choices: [{ text: `Let's eat.`, to: null }] },
          refuse: { text: `Steal? They stole it first. ...No. I see how it is. Principles. Must be nice, being able to afford them.`, choices: [{ text: `(leave)`, to: null }] },
          report: { text: `...Say that again. Slowly. So I can remember the exact shape of a traitor.`, choices: [{ text: `(leave)`, to: null }] },
        },
      },
      ren_warn: {
        speaker: 'Ren of the Ashen Blades', plate: '#9dc08a', start: 's',
        nodes: {
          s: { text: `You're back. What's wrong — you look like you've seen a ghost.`,
            choices: [
              { text: `Gendo wants you dead. His guard is coming — tonight.`, to: 'w',
                do: (c) => S.event3Warn(c) },
              { text: `Nothing. Forget it.`, to: null },
            ] },
          w: { text: `...I see. Then we don't run — we never run. If the clan wants the Ashen Blades, they'll find us sharpened.\n\nStand with us, brother. One more time.`, choices: [{ text: `To the end.`, to: null }] },
        },
      },
      tsubaki: {
        speaker: 'Tsubaki', plate: '#9dc08a', start: 's',
        nodes: {
          s: { text: (c) => S.flags.event2done === 'report'
              ? `Ren told me what you did. We were starving and you sold us for clan silver. I hope it spends well, traitor.`
              : `You're the new blade everyone's whispering about. Word of advice, stray: in Kirisato, everyone wants to use you. The trick is deciding who you let.`,
            choices: [{ text: `Noted.`, to: null }] },
        },
      },
      gendo: {
        speaker: 'Magistrate Gendo', plate: '#e08080', start: 's',
        nodes: {
          s: { text: (c) => {
              if (!S.flags.event2done) return `The magistrate does not receive strays without cause. Captain Isamu handles... lesser matters. Return when you've made yourself useful to the clan.`;
              if (S.flags.event3done) return `Our business is concluded, rōnin. Enjoy the peace you've purchased.`;
              return `The Ashen Blades grow bold, rōnin. My captain speaks of you — a blade with no master and no loyalties.\n\nThat can be... useful.\n\nHunt down their nest in the west alley. Bring me Ren's headband, and the clan will remember its friends.`;
            },
            choices: [
              { text: `I'll hunt them.`, to: 'accept', if: () => S.flags.event2done && !S.flags.event3done,
                do: (c) => S.event3Choice('clan', c) },
              { text: `I won't be your executioner.`, to: 'refuse', if: () => S.flags.event2done && !S.flags.event3done,
                do: (c) => S.event3Choice('refuse', c) },
              { text: `I'll consider it.`, to: 'warn', if: () => S.flags.event2done && !S.flags.event3done,
                do: (c) => S.event3Choice('warn', c) },
              { text: `(leave)`, to: null },
            ] },
          accept: { text: `Good. No speeches, no honor — just results. The west alley. Tonight.`, choices: [{ text: `It'll be done.`, to: null }] },
          refuse: { text: `...Everyone has a price, rōnin. Yours is simply "no." How... expensive.\n\nGet out of my sight. The north road is that way — use it.`, choices: [{ text: `(leave)`, to: null }] },
          warn: { text: `Consider quickly, then. Opportunities rot faster than rice in this town.`, choices: [{ text: `(leave)`, to: null }] },
        },
      },
      sergeant: {
        speaker: 'Tax Sergeant', plate: '#e08080', start: 's',
        nodes: {
          s: { text: `You! Rōnin! This merchant's short on her protection tax — again.\n\nYou look like someone who understands how the world works.`,
            choices: [
              { text: `Leave her alone.`, to: 'fight', do: (c) => S.event1Choice('intervene', c) },
              { text: `She owes the clan. I'll help you collect.`, to: 'side', do: (c) => S.event1Choice('side', c) },
              { text: `(Walk away.)`, to: 'walk', do: (c) => S.event1Choice('walk', c) },
            ] },
          fight: { text: `Big mistake, stray. BOYS!`, choices: [{ text: `(draw your sword)`, to: null }] },
          side: { text: `Ha! A practical one. Hold her still, then — the magistrate rewards cooperation.`, choices: [{ text: `(Mei looks at you with pure contempt.)`, to: null }] },
          walk: { text: `Smart dog. Keep walking.`, choices: [{ text: `(leave)`, to: null }] },
        },
      },
      // v2: riverside docks district — side stories, small rep rewards, no main-story impact
      souta: {
        speaker: 'Fisherman Souta', plate: '#e8c96a', start: 's',
        nodes: {
          s: { text: `Evening, stray. Mind the nets — ...you're not with the harbor inspectors, are you? They took my boat this morning. "Harbor tax," they said. A fisherman without a boat is just a hungry man with opinions.`,
            choices: [
              { text: `I'll pay the harbor tax for you.`, to: 'paid', if: () => !S.flags.sideSouta,
                do: (c) => { S.flags.sideSouta = 'paid'; S.adjust('town', 10); c.toast('Souta gets his boat back.'); c.save(); } },
              { text: `Point me at the inspector's men.`, to: 'fight', if: () => !S.flags.sideSouta,
                do: (c) => S.sideSoutaFight(c) },
              { text: `Hana's missing a crate — seen it?`, to: 'crate',
                if: () => S.flags.sideHana === 'search' && !S.flags.foundCrate,
                do: (c) => { S.flags.foundCrate = true; } },
              { text: `Not my problem, old man.`, to: null },
            ] },
          paid: { text: `You'd do that? ...The sea owes you one, friend. And so do I. The boat's mine again by morning.`, choices: [{ text: `(nod)`, to: null }] },
          fight: { text: `Kano's thug loiters by the warehouse — big fellow, bigger mouth. Take your anger out on him, not me.`, choices: [{ text: `(draw your sword)`, to: null }] },
          crate: { text: `Aye — washed up by the east pilings yesterday, sitting by the third post. Tell Hana it's safe.`, choices: [{ text: `Thanks, Souta.`, to: null }] },
        },
      },
      souta_done: {
        speaker: 'Fisherman Souta', plate: '#e8c96a', start: 's',
        nodes: {
          s: { text: `Boat's mine again, and the nets are full. The sea provides — and so do friends, it turns out.`, choices: [{ text: `(smile)`, to: null }] },
        },
      },
      hana: {
        speaker: 'Hana the Dockworker', plate: '#e8c96a', start: 's',
        nodes: {
          s: { text: (c) => {
              if (!S.flags.sideHana) return `You lift things, stranger? A crate of mine went missing off the far dock — medicine for my mother, not worth stealing, worth everything to me.`;
              if (S.flags.foundCrate) return `You found it? By the east pilings? Oh, thank the river!`;
              return `Any sign of my crate, stranger? Medicine for my mother — not worth stealing, worth everything to me.`;
            },
            choices: [
              { text: `I'll keep an eye out.`, to: 'search', if: () => !S.flags.sideHana,
                do: (c) => { S.flags.sideHana = 'search'; c.toast('Ask around the docks.'); } },
              { text: `I found your crate by the east pilings.`, to: 'found',
                if: () => S.flags.sideHana === 'search' && S.flags.foundCrate,
                do: (c) => { S.flags.sideHana = 'done'; S.adjust('town', 12); c.toast('Hana has her mother\u2019s medicine.'); c.save(); } },
              { text: `Tough luck.`, to: null, if: () => !S.flags.sideHana },
              { text: `Take care, Hana.`, to: null, if: () => !!S.flags.sideHana && S.flags.sideHana !== 'done' },
            ] },
          search: { text: `Bless you. Souta mends his nets down by the east pilings — that old crow sees everything on this river.`, choices: [{ text: `I'll ask him.`, to: null }] },
          found: { text: `That's it! That's her medicine! ...You're a good soul, stray. The docks won't forget this.`, choices: [{ text: `(smile)`, to: null }] },
        },
      },
      hana_done: {
        speaker: 'Hana the Dockworker', plate: '#e8c96a', start: 's',
        nodes: {
          s: { text: `Mother's recovering. I tell everyone on these docks: we've got a guardian with a sword.`, choices: [{ text: `Take care of her.`, to: null }] },
        },
      },
      kage: {
        speaker: 'Kage', plate: '#9dc08a', start: 's',
        nodes: {
          s: { text: `Well well. A blade with no master and no questions. My crew unloads... uninspected goods tonight. You could look the other way. For a consideration.`,
            choices: [
              { text: `Unload it. I'll look away.`, to: 'blind', if: () => !S.flags.sideKage,
                do: (c) => { S.flags.sideKage = 'blind'; S.adjust('rebel', 10); c.toast('The night crew unloads undisturbed.'); c.save(); } },
              { text: `The clan should hear about this.`, to: 'report', if: () => !S.flags.sideKage,
                do: (c) => S.sideKageReport(c) },
              { text: `(Walk away.)`, to: null },
            ] },
          blind: { text: `Pleasure doing business. The night is full of blind eyes — yours are the kindest.`, choices: [{ text: `(leave)`, to: null }] },
          report: { text: `Then you'll die informed. BOYS!`, choices: [{ text: `(draw your sword)`, to: null }] },
        },
      },
      kage_done: {
        speaker: 'Kage', plate: '#9dc08a', start: 's',
        nodes: {
          s: { text: (c) => S.flags.sideKage === 'blind'
              ? `Back for another arrangement? The river provides, friend. The river provides.`
              : `You sold us out to Kano's dogs. Lucky for you I'm a businessman — grudges are bad for trade. Stay out of my sight.`,
            choices: [{ text: `(leave)`, to: null }] },
        },
      },
      kano: {
        speaker: 'Inspector Kano', plate: '#e08080', start: 's',
        nodes: {
          s: { text: `You there. This harbor operates under clan authority. State your business — and mind the cargo.`,
            choices: [
              { text: `The docks run smoother under clan order.`, to: 'order', if: () => !S.flags.sideKano,
                do: (c) => { S.flags.sideKano = 'order'; S.adjust('clan', 10); c.toast('Kano notes your loyalty.'); c.save(); } },
              { text: `Your taxes are strangling this harbor.`, to: 'defy', if: () => !S.flags.sideKano,
                do: (c) => { S.flags.sideKano = 'defy'; S.adjust('clan', -10); S.adjust('town', 5); c.save(); } },
              { text: `Just passing through.`, to: null },
            ] },
          order: { text: `A sensible blade. The magistrate rewards those who appreciate order. Remember that.`, choices: [{ text: `(leave)`, to: null }] },
          defy: { text: `Careful, rōnin. Words like that get remembered. ...As do the men who say them.`, choices: [{ text: `(leave)`, to: null }] },
        },
      },
      kano_done: {
        speaker: 'Inspector Kano', plate: '#e08080', start: 's',
        nodes: {
          s: { text: `Back again? The harbor is orderly. See that it stays that way.`, choices: [{ text: `(leave)`, to: null }] },
        },
      },
    };
  }

  jiroHint() {
    const f = this.flags;
    if (!f.event1done) return `Trouble usually starts at the market, west side. If the tax men are about their business, you'll hear it before you see it.`;
    if (!f.event2done) return `A rebel named Ren skulks in the west alley. Whatever he asks you — think about who you want to be when you answer.`;
    if (!f.event3done) return `The magistrate has been asking after you. The manor, north side. Mind your tongue in there — his patience is thinner than his wine.`;
    return `Whatever you chose... it's done. The town will live with it. So will you.`;
  }

  treeFor(npc) {
    const id = npc.id, f = this.flags;
    if (id === 'jiro') return this.trees[f.introDone ? 'jiro_idle' : 'jiro_intro'];
    if (id === 'ren' && f.event3done === 'warn' && !f.warned) return this.trees['ren_warn'];
    // v2: docks NPCs — side stories resolve to idle trees once done
    if (id === 'souta') {
      if (f.sideHana === 'search' && !f.foundCrate) return this.trees['souta']; // keep the crate lead reachable
      return this.trees[f.sideSouta ? 'souta_done' : 'souta'];
    }
    if (id === 'hana') return this.trees[f.sideHana === 'done' ? 'hana_done' : 'hana'];
    if (id === 'kage') return this.trees[f.sideKage ? 'kage_done' : 'kage'];
    if (id === 'kano') return this.trees[f.sideKano ? 'kano_done' : 'kano'];
    return this.trees[npc.def.tree];
  }

  // ================= EVENTS =================

  event1Choice(which, c) {
    this.flags.event1done = true;
    c.clearActors();
    if (which === 'intervene') {
      this.adjust('rebel', 25); this.adjust('clan', -20); this.adjust('town', 15);
      c.spawnEnemies([
        { type: 'ashigaru', x: -18, z: 8, faction: 'clan' },
        { type: 'ashigaru', x: -22, z: 3, faction: 'clan' },
      ], 'event1');
      c.banner('DRAW YOUR SWORD');
      c.setObjective('Drive off the clan tax collectors');
      c.toast('The guards draw their blades!');
    } else if (which === 'side') {
      this.adjust('clan', 25); this.adjust('town', -15); this.adjust('rebel', -10);
      c.setObjective('The town watches — decide who you are');
      c.toast('Mei will remember this.');
    } else {
      this.adjust('town', -10);
      c.setObjective('The town watches — decide who you are');
    }
    c.save();
  }

  event2Choice(which, c) {
    this.flags.event2done = which;
    if (which === 'accept') {
      this.adjust('rebel', 25); this.adjust('clan', -25);
      c.setObjective('Go to the clan rice storehouse (east)');
      c.banner('THE STOREHOUSE');
      c.toast('Ren slips you a torch. Burn it all.');
    } else if (which === 'refuse') {
      this.adjust('rebel', -10); this.adjust('town', 5);
      c.setObjective('The magistrate seeks an audience (north manor)');
    } else { // report
      this.adjust('clan', 25); this.adjust('rebel', -30);
      c.hideNpc('tsubaki');
      c.spawnEnemies([
        { type: 'rusher', x: -24, z: -8, faction: 'rebel', name: 'Tsubaki', hpMul: 1.6, drops: 'crane' },
        { type: 'rusher', x: -29, z: -13, faction: 'rebel' },
      ], 'event2report');
      c.banner('AMBUSH');
      c.setObjective('Survive the Ashen Blades\u2019 ambush');
      c.toast('You hear blades leaving sheaths behind you...');
    }
    c.save();
  }

  // proximity trigger: player near the storehouse after accepting
  maybeTriggerStorehouseAmbush(playerPos, c) {
    if (this.flags.event2done !== 'accept' || this.flags.storehouseAmbush) return;
    const dx = playerPos.x - 30, dz = playerPos.z - (-6);
    if (dx * dx + dz * dz < 16 * 16) {
      this.flags.storehouseAmbush = true;
      c.spawnEnemies([
        { type: 'ashigaru', x: 26, z: -3, faction: 'clan' },
        { type: 'ashigaru', x: 33, z: -9, faction: 'clan' },
        { type: 'ashigaru', x: 30, z: 0, faction: 'clan' },
      ], 'event2accept');
      c.banner('STOREHOUSE GUARDS');
      c.setObjective('Fight through the storehouse guards');
      c.toast('Arson was never going to be quiet.');
    }
  }

  event3Choice(which, c) {
    this.flags.event3done = which;
    if (which === 'clan') {
      this.adjust('clan', 30); this.adjust('rebel', -30);
      c.hideNpc('ren');
      c.spawnEnemies([
        { type: 'rusher', x: -25, z: -10, faction: 'rebel', name: 'Ren', hpMul: 2.2, drops: 'ember' },
        { type: 'rusher', x: -28, z: -14, faction: 'rebel' },
        { type: 'rusher', x: -22, z: -14, faction: 'rebel' },
      ], 'event3clan');
      c.banner('THE WEST ALLEY');
      c.setObjective('Destroy the Ashen Blades');
    } else if (which === 'refuse') {
      this.adjust('clan', -20);
      c.setObjective('Leave Kirisato by the north road');
      c.banner('EXILE');
    } else { // warn — player must go warn Ren
      this.adjust('rebel', 25); this.adjust('clan', -30);
      c.setObjective('Warn Ren in the west alley');
      c.banner('A WARNING');
    }
    c.save();
  }

  event3Warn(c) {
    this.flags.warned = true;
    c.hideNpc('ren'); c.hideNpc('tsubaki');
    c.spawnEnemies([
      { type: 'hatamoto', x: -24, z: -8, faction: 'clan', name: 'Captain Isamu', hpMul: 1.3, drops: 'moon' },
      { type: 'ashigaru', x: -27, z: -12, faction: 'clan' },
      { type: 'ashigaru', x: -21, z: -13, faction: 'clan' },
    ], 'event3warn');
    c.banner('THE CLAN STRIKES');
    c.setObjective('Stand with the Ashen Blades — survive');
  }

  // v2: docks side stories with combat
  sideSoutaFight(c) {
    this.flags.sideSouta = 'fight';
    this.adjust('clan', -10); this.adjust('town', 5);
    c.spawnEnemies([
      { type: 'ashigaru', x: 42, z: -16, faction: 'clan', name: 'Dock Enforcer' },
    ], 'sidesouta');
    c.banner('DRAW YOUR SWORD');
    c.setObjective('Drive off the dock enforcer');
    c.save();
  }

  sideKageReport(c) {
    this.flags.sideKage = 'report';
    this.adjust('clan', 10); this.adjust('rebel', -15);
    c.spawnEnemies([
      { type: 'rusher', x: 42, z: -8, faction: 'rebel' },
      { type: 'rusher', x: 46, z: -14, faction: 'rebel' },
      { type: 'hatamoto', x: 42.5, z: -13.5, faction: 'rebel', name: `Kage's Bodyguard`, hpMul: 1.4, drops: 'willow' },
    ], 'sidekage');
    c.banner('AMBUSH');
    c.setObjective('Survive the smugglers\u2019 ambush');
    c.save();
  }

  // v2: dusk -> night once event 2 is resolved (lerped over ~10s by the world)
  toNight(c) {
    if (!this.flags.night) { this.flags.night = true; c.setNight(); }
  }

  onEnemiesCleared(tag, c) {
    if (tag === 'event1') {
      c.setObjective('The market is safe — for now. Find Ren in the west alley');
      c.toast('The market breathes again.');
      this.adjust('town', 5);
      c.save();
    } else if (tag === 'event2accept') {
      c.setObjective('The storehouse is yours — report to Ren, or see the magistrate');
      c.toast('The storehouse burns. The town will eat this winter.');
      this.adjust('rebel', 10); this.adjust('town', 10);
      this.toNight(c); c.save();
    } else if (tag === 'event2report') {
      c.setObjective('The magistrate seeks an audience (north manor)');
      c.toast('The alley falls silent.');
      this.toNight(c); c.save();
    } else if (tag === 'sidesouta') {
      c.setObjective('The docks breathe easier');
      c.toast('Souta will get his boat back.');
      this.adjust('town', 5);
      c.save();
    } else if (tag === 'sidekage') {
      c.setObjective('The smugglers scatter into the dark');
      c.toast('Kano\u2019s harbor is quiet — for now.');
      this.adjust('clan', 5);
      c.save();
    } else if (tag === 'event3clan' || tag === 'event3warn') {
      this.evaluateEnding(c);
    }
  }

  // north-road trigger for the refuse path
  maybeTriggerExile(playerPos, c) {
    if (this.flags.event3done !== 'refuse' || this.ended) return;
    if (playerPos.z < -44 && Math.abs(playerPos.x) < 14) this.evaluateEnding(c);
  }

  evaluateEnding(c) {
    if (this.ended) return;
    this.ended = true;
    const r = this.rep, k = this.kills;
    let key;
    // The Butcher: cut down fighters from both sides yet belong to neither —
    // betrayal and blood over loyalty.
    if (k.clan + k.rebel >= 7 && r.clan < 50 && r.rebel < 50) key = 'butcher';
    else if (r.clan >= 50) key = 'clan';
    else if (r.rebel >= 50) key = 'rebel';
    else if (r.town >= 35) key = 'town';
    else key = 'wolf';
    c.endGame(key, { rep: { ...r }, kills: { ...k } });
  }

  onDeath(c) {
    if (this.ended) return;
    this.ended = true;
    c.endGame('death', { rep: { ...this.rep }, kills: { ...this.kills } });
  }
}

function label(f) {
  return f === 'clan' ? 'CLAN REP' : f === 'rebel' ? 'REBEL REP' : 'TOWN REP';
}
