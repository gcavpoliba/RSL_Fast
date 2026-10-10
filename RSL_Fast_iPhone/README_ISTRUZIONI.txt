================================================================
 RSL FAST — RISPOSTA SISMICA 1D
 Pacchetto per installare l'app su iPhone (PWA)
================================================================

COS'E' QUESTO PACCHETTO?
------------------------
Questo archivio contiene l'applicazione web completa (HTML + JS +
icone) di RSL Fast — Analisi di risposta sismica locale 1D con
metodo lineare equivalente (EERA/STRATA), il porting per iPhone
dell'applicazione Python originale gcavpoliba/RSL_Fast.

Tutto il motore di calcolo gira DENTRO l'iPhone (JavaScript
lato client): nessun dato viene inviato a nessun server.

PERCHE' NON ESISTE UN FILE ".IPA" DA TOCCARE PER INSTALLARE?
-----------------------------------------------------------
Apple non consente di installare app da file (come gli .apk di
Android). Un'app nativa installabile richiede obbligatoriamente:
  - un Mac con Xcode per compilare,
  - un account Apple Developer (99 €/anno) per firmare l'app,
  - la distribuzione via App Store o TestFlight.
La via UFFICIALE Apple per "installare" un software web su iPhone
e' la PWA (Progressive Web App): si aggiunge alla schermata Home
da Safari e funziona come un'app normale, a schermo intero, con
la sua icona, anche OFFLINE.

COME INSTALLARE SU IPHONE (5 MINUTI, GRATIS)
--------------------------------------------
PASSO 1 — Pubblica l'app online (per avere un indirizzo https):
  a) Estrai questo archivio sul computer.
  b) Vai su https://app.netlify.com/drop (account gratuito)
     oppure https://dash.cloudflare.com -> Workers & Pages.
  c) TRASCINA la cartella estratta nella pagina.
  d) Dopo pochi secondi hai un indirizzo tipo:
     https://rsl-fast.netlify.app

  In alternativa con GitHub (hai gia' l'account):
  a) Crea un nuovo repository (es. rsl-fast-app), branch "gh-pages".
  b) Trascina i file estratti nella pagina di upload del branch.
  c) Impostazioni -> Pages -> Source: branch "gh-pages".
  d) Indirizzo: https://gcavpoliba.github.io/rsl-fast-app/
     (NOTA: con GitHub Pages serve aggiungere "basePath" nella
     configurazione e ricompilare; Netlify/Cloudflare sono piu'
     semplici per questo pacchetto.)

PASSO 2 — Installa sull'iPhone:
  1. Apri SAFARI sull'iPhone (non Chrome).
  2. Vai all'indirizzo pubblicato.
  3. Tocca il pulsante CONDIVIDI (quadrato con freccia in su).
  4. Scorri e tocca "AGGIUNGI ALLA HOME".
  5. Tocca "Aggiungi".
  -> L'icona "RSL Fast" appare in Home: si apre a schermo intero
     come un'app nativa, funziona anche SENZA internet (dopo il
     primo avvio) e salva i progetti sul dispositivo.

FILE INCLUSI
------------
  index.html             pagina dell'app
  _next/                 JavaScript e CSS dell'app
  manifest.json          manifest PWA (icona, nome, modalita' standalone)
  sw.js                  service worker (funzionamento offline)
  icons/                 icone dell'app (192/512/maskable/apple-touch 180)
  robots.txt
  RSL_Fast_offline.html  BONUS: l'intera app in UN SOLO file HTML.
                         Non serve pubblicarlo: puoi aprirlo diretta-
                         mente (browser, email, chiavetta). Comodo come
                         copia di riserva, ma per installarlo in Home
                         sull'iPhone usa il sito pubblicato (PASSO 1+2).

REQUISITI
---------
  iPhone con iOS 14 o successivo (ottimo su iPhone 16 / iOS 18+).
  Hosting statico qualsiasi (Netlify, Cloudflare Pages, GitHub
  Pages, Vercel) con HTTPS — tutti gratuiti per questo uso.

NOTE TECNICHE
-------------
  - Motore di calcolo validato contro il codice Python originale
    (precisione 1e-13 o migliore su tutte le grandezze).
  - I progetti (.json) si salvano/riaprono dall'app; i dati
    restano nel localStorage del dispositivo.
  - Il service worker usa strategia network-first: aggiorni
    l'app semplicemente ripubblicando la cartella.
================================================================
